# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
import math

class InfoNCELoss(nn.Module):
    def __init__(self, init_temp: float = 0.07, lambda_sup: float = 0.3, 
                 hard_negative_weight: float = 0.75, max_temp: float = 100.0):
        """
        Args:
            init_temp: Initial temperature value (default 0.07, consistent with CLIP)
            lambda_sup: Weight of supervised loss (between 0-1)
            hard_negative_weight: Hard negative weighting coefficient (0=not used, recommended 0.5-1.0)
            max_temp: Temperature upper bound constraint (prevent temperature from diverging and causing model failure)
        """
        super().__init__()
        self.lambda_sup = lambda_sup
        self.hard_negative_weight = hard_negative_weight
        self.max_temp = max_temp
        
        # Learnable temperature parameter: Use exponential parameterization logit_scale = exp(logit_scale_param)
        # Thus temperature = 1 / logit_scale = exp(-logit_scale_param)
        # Initial temperature 0.07 -> logit_scale ≈ 14.28 -> log(14.28) ≈ 2.66
        init_logit_scale = math.log(1.0 / init_temp)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        
    def get_temperature(self):
        """Get current temperature value (for logging and debugging)"""
        # Constrain temperature range to prevent divergence during optimization
        logit_scale_clamped = torch.clamp(self.logit_scale, math.log(1.0 / self.max_temp), 100)
        return 1.0 / logit_scale_clamped.exp()
    
    def get_logit_scale(self):
        """Get current logit_scale (for computation)"""
        logit_scale_clamped = torch.clamp(self.logit_scale, math.log(1.0 / self.max_temp), 100)
        return logit_scale_clamped.exp()
        
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        feat1, feat2: Normalized features, both with shape [batch, embed_dim]
        labels: Category labels, shape [batch]
        """
        # Auto handle multi-GPU gathering
        if dist.is_initialized():
            feat1 = self._all_gather(feat1)
            feat2 = self._all_gather(feat2)
            labels = self._all_gather(labels)
        
        batch_size = feat1.shape[0]
        device = feat1.device
        
        # Get current learnable logit_scale
        logit_scale = self.get_logit_scale()
        
        # 1. Compute standard InfoNCE loss (bidirectional cross-entropy) - using learnable logit_scale
        logits = torch.matmul(feat1, feat2.T) * logit_scale  # Note: multiply by logit_scale, not divide by temperature
        labels_diag = torch.arange(batch_size, device=device)
        
        loss_i2t = F.cross_entropy(logits, labels_diag, reduction='none')
        loss_t2i = F.cross_entropy(logits.T, labels_diag, reduction='none')
        
        # Hard negative weighting (core addition)
        if self.hard_negative_weight > 0:
            # Compute difficulty for each sample (higher loss = harder)
            with torch.no_grad():
                # Sample difficulty scores
                hardness_i2t = loss_i2t / (loss_i2t.mean() + 1e-8)
                hardness_t2i = loss_t2i / (loss_t2i.mean() + 1e-8)
                
                # Weighting weights
                weights_i2t = 1.0 + self.hard_negative_weight * hardness_i2t
                weights_t2i = 1.0 + self.hard_negative_weight * hardness_t2i
            
            # Apply weights
            loss_i2t = (loss_i2t * weights_i2t).mean()
            loss_t2i = (loss_t2i * weights_t2i).mean()
        else:
            loss_i2t = loss_i2t.mean()
            loss_t2i = loss_t2i.mean()
        
        base_loss = (loss_i2t + loss_t2i) / 2
        
        # 2. Supervised contrastive loss (fixed numerical stability issue)
        labels_eq = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float()
        sim_matrix = logits
        
        pos_mask = labels_eq.bool()
        if pos_mask.sum() > batch_size:  # Ensure at least one other same-class sample exists
            # Compute average similarity of same-class samples (excluding self)
            pos_sim = (sim_matrix * labels_eq).sum(dim=1) - torch.diag(sim_matrix)
            pos_count = labels_eq.sum(dim=1) - 1  # Exclude self
            pos_sim = pos_sim / (pos_count + 1e-8)
            
            # Compute global average similarity of all samples
            all_sim = sim_matrix.mean(dim=1)
            
            gap = pos_sim - all_sim           
            sup_loss = (torch.exp(torch.relu(-gap)) - 1).mean()
        else:
            sup_loss = torch.tensor(0.0, device=device)
        
        total_loss = base_loss + self.lambda_sup * sup_loss
        
        return total_loss
    
    @staticmethod
    def _all_gather(tensor: torch.Tensor) -> torch.Tensor:
        """Inline all_gather implementation, skip gather for scalars to avoid warning"""
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        
        world_size = dist.get_world_size()
        if world_size == 1:
            return tensor
        
        # Key modification: Return scalar directly (all processes have same value, no sync needed)
        if tensor.dim() == 0:
            return tensor
        
        tensor = tensor.contiguous()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        gathered[dist.get_rank()] = tensor
        
        return torch.cat(gathered, dim=0)