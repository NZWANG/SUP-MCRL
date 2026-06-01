"""
Image codebook module - Integrated optimized version (fixed version)
"""
#PPA
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class ImageCodebook(nn.Module):
    """
    Hierarchical learnable image codebook - Integrated improved version
    """
    def __init__(self, dim: int, 
                 num_codewords_l1: int = 64,
                 num_codewords_l2: int = 128,
                 num_codewords_l3: int = 320,
                 k: int = 16, 
                 dropout: float = 0.1, 
                 max_residual_strength: float = 0.2,
                 num_experts: int = 4,
                 ema_decay: float = 0.99):
        super().__init__()
        self.dim = dim
        self.k = k
        self.num_experts = num_experts
        self.ema_decay = ema_decay
        
        self.num_codewords_l1 = num_codewords_l1
        self.num_codewords_l2 = num_codewords_l2
        self.num_codewords_l3 = num_codewords_l3
        
        self.codebook_l1 = nn.Parameter(torch.randn(num_codewords_l1, dim))
        self.codebook_l2 = nn.Parameter(torch.randn(num_codewords_l2, dim))
        self.codebook_l3 = nn.Parameter(torch.randn(num_codewords_l3, dim))

        # Unified initialization: Normal projection + lightweight repulsion optimization
        for cb in [self.codebook_l1, self.codebook_l2, self.codebook_l3]:
            self._init_codebook(cb)

        '''
        orthogonal_init = True
        for cb in [self.codebook_l1, self.codebook_l2, self.codebook_l3]:
            nn.init.xavier_uniform_(cb, gain=1.0)
            if orthogonal_init:
                with torch.no_grad():
                    n, d = cb.shape
                    if n <= d:
                        q, r = torch.linalg.qr(cb.T, mode='reduced')
                        cb.copy_(q.T)
                    else:
                        u, s, vh = torch.linalg.svd(cb, full_matrices=False)
                        cb[:d].copy_(vh)
                        if n > d:
                            remaining = torch.randn(n - d, d, device=cb.device, dtype=cb.dtype)
                            base = cb[:d]
                            proj = remaining @ base.T @ base
                            remaining = remaining - proj
                            remaining = remaining / (remaining.norm(dim=1, keepdim=True) + 1e-8)
                            cb[d:].copy_(remaining)
        
        self.scale_l1 = nn.Parameter(torch.ones(1) * math.sqrt(dim))  # FIX: Initialize with larger temperature to avoid overly sharp distribution
        self.scale_l2 = nn.Parameter(torch.ones(1) * math.sqrt(dim))
        self.scale_l3 = nn.Parameter(torch.ones(1) * math.sqrt(dim))
        '''

        self.scale_l1 = nn.Parameter(torch.ones(1))  # FIX: Initialize with larger temperature to avoid overly sharp distribution
        self.scale_l2 = nn.Parameter(torch.ones(1))
        self.scale_l3 = nn.Parameter(torch.ones(1))
        
        self.max_residual_strength = max_residual_strength
        self.residual_gate_l1 = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.SiLU(),
            nn.Linear(dim // 4, 1), nn.Sigmoid()
        )
        self.residual_gate_l2 = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.SiLU(),
            nn.Linear(dim // 4, 1), nn.Sigmoid()
        )
        self.residual_gate_l3 = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.SiLU(),
            nn.Linear(dim // 4, 1), nn.Sigmoid()
        )
        
        self.router = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_experts, bias=False)
        )
        
        assert num_codewords_l1 % num_experts == 0
        assert num_codewords_l2 % num_experts == 0
        assert num_codewords_l3 % num_experts == 0
        
        self.register_buffer('ema_codebook_l1', torch.zeros(num_codewords_l1, dim))
        self.register_buffer('ema_codebook_l2', torch.zeros(num_codewords_l2, dim))
        self.register_buffer('ema_codebook_l3', torch.zeros(num_codewords_l3, dim))
        self.register_buffer('ema_initialized', torch.tensor(False))
        
        self.cross_attn_l1 = nn.MultiheadAttention(
            embed_dim=dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.cross_attn_l2 = nn.MultiheadAttention(
            embed_dim=dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.cross_attn_l3 = nn.MultiheadAttention(
            embed_dim=dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        self.hierarchy_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False)
        )
        
        self._init_ema()

    def _init_codebook(self, cb: torch.Tensor, steps: int = 30):
        """
        Spherical uniform initialization: Normal projection + lightweight repulsion optimization
        Do not enforce orthogonality, pursue maximum minimum angle, ensure all directions are represented
        """
        n, d = cb.shape
        with torch.no_grad():
            # Normal distribution projected to sphere is uniform distribution
            vecs = torch.randn(n, d, device=cb.device, dtype=cb.dtype)
            vecs = F.normalize(vecs, p=2, dim=-1)
            
            # Lightweight repulsion: Eliminate random clustering
            for _ in range(steps):
                dots = torch.matmul(vecs, vecs.t())
                mask = torch.eye(n, device=cb.device, dtype=torch.bool)
                dots = dots.masked_fill(mask, 0)
                
                # Repulsion force: Larger dot product (closer), stronger force
                forces = torch.matmul(dots, vecs)
                proj = (forces * vecs).sum(dim=-1, keepdim=True)
                tangent = forces - proj * vecs
                
                vecs = F.normalize(vecs + 0.1 * tangent, p=2, dim=-1)
            
            cb.copy_(vecs)
    
    def _init_ema(self):
        if not self.ema_initialized:
            self.ema_codebook_l1.copy_(self.codebook_l1.data)
            self.ema_codebook_l2.copy_(self.codebook_l2.data)
            self.ema_codebook_l3.copy_(self.codebook_l3.data)
            self.ema_initialized.fill_(True)
    
    def _hierarchical_select(self, query_norm: torch.Tensor):
        B = query_norm.size(0)
        
        cb_l1_norm = F.normalize(self.codebook_l1, p=2, dim=-1)
        logits_l1 = torch.matmul(query_norm, cb_l1_norm.t()) * self.scale_l1 
        
        router_logits = self.router(query_norm)
        expert_weights = F.softmax(router_logits, dim=-1)
        
        codewords_per_expert_l1 = self.num_codewords_l1 // self.num_experts
        expert_mask_l1 = expert_weights.repeat_interleave(codewords_per_expert_l1, dim=1)
        masked_logits_l1 = logits_l1 + torch.log(expert_mask_l1 + 1e-10)
        
        k1 = max(self.k // 3, 4)
        topk_val_l1, topk_idx_l1 = torch.topk(masked_logits_l1, k1, dim=-1)
        weights_l1 = F.softmax(topk_val_l1, dim=-1)
        
        cb_l2_norm = F.normalize(self.codebook_l2, p=2, dim=-1) 
        logits_l2 = torch.matmul(query_norm, cb_l2_norm.t()) * self.scale_l2
        
        codewords_per_expert_l2 = self.num_codewords_l2 // self.num_experts
        expert_mask_l2 = expert_weights.repeat_interleave(codewords_per_expert_l2, dim=1)
        masked_logits_l2 = logits_l2 + torch.log(expert_mask_l2 + 1e-10)
        
        k2 = max(self.k // 3, 4)
        topk_val_l2, topk_idx_l2 = torch.topk(masked_logits_l2, k2, dim=-1)
        weights_l2 = F.softmax(topk_val_l2, dim=-1)
        
        cb_l3_norm = F.normalize(self.codebook_l3, p=2, dim=-1) 
        logits_l3 = torch.matmul(query_norm, cb_l3_norm.t()) * self.scale_l3 
        
        codewords_per_expert_l3 = self.num_codewords_l3 // self.num_experts
        expert_mask_l3 = expert_weights.repeat_interleave(codewords_per_expert_l3, dim=1)
        masked_logits_l3 = logits_l3 + torch.log(expert_mask_l3 + 1e-10)
        
        k3 = max(self.k - k1 - k2, 4)
        topk_val_l3, topk_idx_l3 = torch.topk(masked_logits_l3, k3, dim=-1)
        weights_l3 = F.softmax(topk_val_l3, dim=-1)
        
        return (topk_idx_l1, weights_l1, logits_l1), (topk_idx_l2, weights_l2, logits_l2), (topk_idx_l3, weights_l3, logits_l3), expert_weights
    
    def _compute_soft_weights(self, logits: torch.Tensor, topk_indices: torch.Tensor, 
                             topk_weights: torch.Tensor, residual_gate: torch.Tensor):
        B, num_codewords = logits.shape
        k = topk_indices.size(1)
        
        soft_weights = F.softmax(logits, dim=-1)
        
        hard_weights = torch.zeros_like(soft_weights)
        hard_weights.scatter_(1, topk_indices, topk_weights)
        
        adaptive_residual = residual_gate * self.max_residual_strength
        
        topk_mask = torch.zeros(B, num_codewords, device=logits.device, dtype=torch.bool)
        topk_mask.scatter_(1, topk_indices, True)
        
        residual_weights = soft_weights * (~topk_mask).float() * adaptive_residual
        
        final_weights = hard_weights + residual_weights
        return final_weights
    
    def _update_ema(self):
        if self.training and self.ema_initialized:
            self.ema_codebook_l1.mul_(self.ema_decay).add_(self.codebook_l1.data, alpha=1-self.ema_decay)
            self.ema_codebook_l2.mul_(self.ema_decay).add_(self.codebook_l2.data, alpha=1-self.ema_decay)
            self.ema_codebook_l3.mul_(self.ema_decay).add_(self.codebook_l3.data, alpha=1-self.ema_decay)
    
    def forward(self, query: torch.Tensor):
        B = query.size(0)
        query_norm = F.normalize(query, p=2, dim=-1)
        
        (idx_l1, w_l1, logits_l1), (idx_l2, w_l2, logits_l2), (idx_l3, w_l3, logits_l3), expert_w = \
            self._hierarchical_select(query_norm)
        
        gate_l1 = self.residual_gate_l1(query)
        gate_l2 = self.residual_gate_l2(query)
        gate_l3 = self.residual_gate_l3(query)
        
        final_w_l1 = self._compute_soft_weights(logits_l1, idx_l1, w_l1, gate_l1)
        final_w_l2 = self._compute_soft_weights(logits_l2, idx_l2, w_l2, gate_l2)
        final_w_l3 = self._compute_soft_weights(logits_l3, idx_l3, w_l3, gate_l3)
        
        # FIX: Also L2 normalize query, consistent with protos scale
        query_attn = F.normalize(query, p=2, dim=-1).unsqueeze(1)
        
        protos_l1 = F.normalize(self.codebook_l1[idx_l1], p=2, dim=-1)
        attn_l1, _ = self.cross_attn_l1(query_attn, protos_l1, protos_l1)
        attn_l1 = attn_l1.squeeze(1)
        
        protos_l2 = F.normalize(self.codebook_l2[idx_l2], p=2, dim=-1) 
        attn_l2, _ = self.cross_attn_l2(query_attn, protos_l2, protos_l2)
        attn_l2 = attn_l2.squeeze(1)
        
        protos_l3 = F.normalize(self.codebook_l3[idx_l3], p=2, dim=-1) 
        attn_l3, _ = self.cross_attn_l3(query_attn, protos_l3, protos_l3)
        attn_l3 = attn_l3.squeeze(1)
        
        concat = torch.cat([attn_l1, attn_l2, attn_l3], dim=-1)
        fused = self.hierarchy_fusion(concat)
        
        enhanced = self.output_proj(fused)
        
        self._update_ema()
        
        weights = {
            'l1': final_w_l1,
            'l2': final_w_l2,
            'l3': final_w_l3,
            'expert': expert_w
        }
        
        return enhanced, weights


class RefineBlock(nn.Module):
    """
    Solution 6: Cross-layer feature reuse (Dense Connection)
    """
    def __init__(self, dim: int, dropout: float, expansion: int = 4):
        super().__init__()
        self.dim = dim
        hidden_dim = dim * expansion
        
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Dropout(dropout)
        )
        
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=4, dropout=dropout, batch_first=True
        )
        # FIX: Use independent LayerNorm for query and key/value respectively to avoid scale asymmetry
        self.cross_scale_norm_q = nn.LayerNorm(dim)
        self.cross_scale_norm_kv = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, prev_features: Optional[torch.Tensor] = None):
        if prev_features is not None:
            x_norm = self.cross_scale_norm_q(x)
            prev_norm = self.cross_scale_norm_kv(prev_features)  # FIX: Also normalize prev_features
            x_reshaped = x_norm.unsqueeze(1)
            prev_reshaped = prev_norm.unsqueeze(1)
            cross_out, _ = self.cross_scale_attn(
                query=x_reshaped,
                key=prev_reshaped,
                value=prev_reshaped
            )
            x = x + cross_out.squeeze(1)
        
        return x + self.ffn(self.norm(x))


class EEGToImageAttention(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2,
                 num_codewords: int = 512,
                 topk: int = 16,
                 dropout: float = 0.3, 
                 expansion: int = 4,
                 img_guidance_ratio: float = 0.3, #0.5
                 max_residual_strength: float = 0.2, #0.2
                 num_experts: int = 4, #4
                 ema_decay: float = 0.99):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.img_guidance_ratio = img_guidance_ratio
        
        self.image_codebook = ImageCodebook(
            dim=dim,
            num_codewords_l1=num_codewords//4,
            num_codewords_l2=num_codewords//2,
            num_codewords_l3=num_codewords,
            k=topk,
            dropout=dropout,
            max_residual_strength=max_residual_strength,
            num_experts=num_experts,
            ema_decay=ema_decay
        )
        
        self.eeg_query_net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.eeg_refine = nn.ModuleList([
            RefineBlock(dim, dropout, expansion) 
            for _ in range(num_layers)
        ])
        
        self.eeg_output_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False)
        )
        
        self.img_query_net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # FIX: Restore LayerNorm in fusion_gate, add independent fusion_norm for current/target standardization
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.fusion_norm = nn.LayerNorm(dim)  # FIX: Replace F.layer_norm, provide learnable affine parameters
        
        self.fusion_alpha = nn.Parameter(torch.tensor(0.3))
        
        self.img_to_eeg_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, eeg_feat: torch.Tensor, img_feat: Optional[torch.Tensor] = None):
        B, D = eeg_feat.shape
        eeg_residual = eeg_feat
        
        eeg_query = self.eeg_query_net(eeg_feat)
        eeg_codebook_out, eeg_weights = self.image_codebook(eeg_query)
        
        eeg_enhanced = eeg_codebook_out
        prev_features = None
        
        for i, layer in enumerate(self.eeg_refine):
            if i == 0:
                eeg_enhanced = layer(eeg_enhanced, None)
            else:
                eeg_enhanced = layer(eeg_enhanced, prev_features)
            
            if i < len(self.eeg_refine) - 1:
                prev_features = eeg_enhanced.clone()
        
        eeg_enhanced = self.eeg_output_proj(eeg_enhanced)
        
        # ========== Test Mode ==========
        if not self.training or img_feat is None:
            output = eeg_residual + torch.sigmoid(self.fusion_alpha) * eeg_enhanced
            return F.normalize(output, p=2, dim=-1)
        
        # ========== Training Mode ==========
        use_img = torch.rand(B, device=eeg_feat.device) < self.img_guidance_ratio
        
        if use_img.any():
            with torch.no_grad():
                img_query = self.img_query_net(img_feat[use_img])
                img_codebook_out, _ = self.image_codebook(img_query)
                target = self.img_to_eeg_proj(img_codebook_out)
            
            current = eeg_enhanced[use_img]
            target_detached = target.detach()
            
            # FIX: Use nn.LayerNorm instead of F.layer_norm, fix normalized_shape
            current = self.fusion_norm(current)
            target_detached = self.fusion_norm(target_detached)
            
            concat = torch.cat([current, target_detached], dim=-1)
            gate = self.fusion_gate(concat)
            
            eeg_enhanced[use_img] = gate * target_detached + (1 - gate) * current
        
        output = eeg_residual + torch.sigmoid(self.fusion_alpha) * eeg_enhanced
        
        # FIX: Consistent train/test output behavior, unified L2 normalization
        return F.normalize(output, p=2, dim=-1)