#SAVE
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSparseDilatedConv(nn.Module):
    """
    Lightweight multi-scale sparse dilated convolution (2-branch simplified version)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 3]):
        super().__init__()
        self.num_branches = len(dilations)
        
        branch_out = out_channels // 2
        self.branch_outs = [branch_out, out_channels - branch_out]
        
        self.branches = nn.ModuleList()
        for i, dilation in enumerate(dilations):
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, self.branch_outs[i], kernel_size, 
                         padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(self.branch_outs[i]),
            ))
        
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.num_branches),
            nn.Softmax(dim=1)
        )
        
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        B = x.shape[0]
        weights = self.weight_predictor(x)
        
        outs = []
        for i, branch in enumerate(self.branches):
            out = branch(x)
            outs.append(out * weights[:, i].view(B, 1, 1, 1))
        
        fused = torch.cat(outs, dim=1)
        fused = self.fusion_conv(fused)
        fused = self.bn(fused)
        return self.relu(fused)


class LightweightMultiScaleEncoder(nn.Module):
    """
    Lightweight multi-scale encoder
    """
    
    def __init__(self, in_channels=3, feature_dim=512):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.ms1 = MultiScaleSparseDilatedConv(64, 64, dilations=[1, 2])
        
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.ms2 = MultiScaleSparseDilatedConv(128, 128, dilations=[1, 3])
        
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.ms3 = MultiScaleSparseDilatedConv(256, 256, dilations=[1, 4])
        
        self.down4 = nn.Sequential(
            nn.Conv2d(256, feature_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.ms4 = MultiScaleSparseDilatedConv(feature_dim, feature_dim, dilations=[1, 3])
        
    def forward(self, x):
        features = []
        
        x = self.stem(x)
        features.append(x)
        
        x = self.down1(x)
        x = self.ms1(x)
        features.append(x)
        
        x = self.down2(x)
        x = self.ms2(x)
        features.append(x)
        
        x = self.down3(x)
        x = self.ms3(x)
        features.append(x)
        
        x = self.down4(x)
        x = self.ms4(x)
        features.append(x)
        
        return features


class CenterPriorModule(nn.Module):
    """
    Center prior module - Hard-coded Gaussian, no parameters
    """
    
    def __init__(self, initial_sigma=0.2, final_sigma=2.5, warmup_epochs=15):
        super().__init__()
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        
    def forward(self, h, w, device):
        if self.current_epoch < self.warmup_epochs:
            alpha = self.current_epoch / self.warmup_epochs
            sigma = self.initial_sigma + (self.final_sigma - self.initial_sigma) * alpha
        else:
            sigma = self.final_sigma
        
        y = torch.arange(h, device=device).float()
        x = torch.arange(w, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        gaussian = torch.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2 * max(h, w)**2))
        
        return gaussian


class MultiScaleSpatialDecoder(nn.Module):
    """
    Lightweight multi-scale spatial decoder - Strong contrast version
    Outputs 3-channel independent weights, achieves hard attention through learnable temperature
    """
    
    def __init__(self, feature_dim=512):
        super().__init__()
        
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(256, 128, 1, bias=False),
            nn.Conv2d(128, 64, 1, bias=False),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.Conv2d(32, 16, 1, bias=False),
        ])
        
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                MultiScaleSparseDilatedConv(feature_dim + 128, 256, dilations=[1, 2]),
            ),
            nn.Sequential(
                MultiScaleSparseDilatedConv(256 + 64, 128, dilations=[1, 2]),
            ),
            nn.Sequential(
                MultiScaleSparseDilatedConv(128 + 32, 64, dilations=[1, 2]),
            ),
            nn.Sequential(
                MultiScaleSparseDilatedConv(64 + 16, 32, dilations=[1, 2]),
            ),
        ])
        
        # Final output layer - directly output logits
        self.final_up = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1, bias=True),
        )
        
        self.center_prior = CenterPriorModule(initial_sigma=0.2, final_sigma=2.5, warmup_epochs=15)
        self.prior_gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, features, target_size, epoch=None):
        if epoch is not None:
            self.center_prior.set_epoch(epoch)
        
        x = features[-1]
        skip_indices = [3, 2, 1, 0]
        
        for i, (block, skip_idx) in enumerate(zip(self.up_blocks, skip_indices)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            skip = self.skip_projs[i](features[skip_idx])
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        network_logits = self.final_up(x)
        
        B, _, H, W = network_logits.shape
        prior_mask = self.center_prior(H, W, network_logits.device)
        prior_mask = prior_mask.view(1, 1, H, W).expand(B, 3, H, W)
        
        gate = torch.sigmoid(self.prior_gate)
        weight_logits = network_logits + gate * torch.log(prior_mask + 1e-8)
        
        return weight_logits


class CompensatedFeatureCompressor(nn.Module):
    """
    Compensation-enhanced feature compressor
    """
    
    def __init__(self, feature_dim=512, refine_ratio=0.1):
        super().__init__()
        self.refine_ratio = refine_ratio
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.refine_mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim),
        )
        
        self.norm = nn.BatchNorm1d(feature_dim)
        
    def forward(self, x):
        B = x.shape[0]
        
        base_feat = self.gap(x).view(B, -1)
        refine = self.refine_mlp(base_feat)
        feat = base_feat + self.refine_ratio * refine
        
        feat = self.norm(feat)
        feat = F.normalize(feat, p=2, dim=1)
        
        return feat


class DilatedSubjectWeighting(nn.Module):
    """
    Multi-scale sparse dilated convolution subject weighting network (strong contrast version)
    
    Core design:
    1. Hard attention mechanism: Push weights toward extremes through learnable temperature
    2. Subject enhancement coefficient > 1.0, background suppression coefficient < 1.0
    3. Center prior guides subject localization
    4. 3-channel independent, color fidelity
    
    Outputs:
    - visual_output: (B, 3, H, W) Subject-enhanced, background faded/removed image
    - feature_output: (B, 512) L2-normalized features
    """
    
    def __init__(self, input_size=224, feature_dim=512, refine_ratio=0.1):
        super().__init__()
        self.input_size = input_size
        self.feature_dim = feature_dim
        
        self.encoder = LightweightMultiScaleEncoder(3, feature_dim)
        self.spatial_decoder = MultiScaleSpatialDecoder(feature_dim)
        self.feature_compressor = CompensatedFeatureCompressor(feature_dim, refine_ratio)
        
        # Hard attention temperature - Controls weight polarization degree
        # Smaller value = weights closer to binary (subject>1, background<<1)
        self.attn_temp = nn.Parameter(torch.ones(1) * 0.3)
        
        # Subject enhancement magnitude - Learnable, initial 1.5x
        self.enhance_scale = nn.Parameter(torch.ones(1) * 0.4)
        
        # Background suppression magnitude - Learnable, initial 0.3x
        self.suppress_scale = nn.Parameter(torch.ones(1) * -1.2)
        
        # Global contrast control
        self.contrast_boost = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.feature_compressor.refine_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.spatial_decoder.final_up.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        nn.init.constant_(self.spatial_decoder.prior_gate, -2.0)
        
        # Hard attention parameter initialization
        nn.init.constant_(self.attn_temp, 0.3)
        nn.init.constant_(self.enhance_scale, 0.4)
        nn.init.constant_(self.suppress_scale, -1.2)
        nn.init.constant_(self.contrast_boost, 0.0)
    
    def forward(self, x, epoch=None):
        B, C, H, W = x.shape
        original_size = (H, W)
        
        if H != self.input_size or W != self.input_size:
            x_resized = F.interpolate(x, size=(self.input_size, self.input_size),
                                    mode='bilinear', align_corners=False)
        else:
            x_resized = x
        
        features = self.encoder(x_resized)
        weight_logits = self.spatial_decoder(features, target_size=(self.input_size, self.input_size), epoch=epoch)
        
        # === Hard Attention Mechanism ===
        # Temperature scaling: Smaller temp = sharper distribution
        temp = F.softplus(self.attn_temp) + 0.05  # Ensure lower bound 0.05 to avoid division by zero
        
        # Temperature-scale logits then sigmoid, push toward 0 or 1
        attn_base = torch.sigmoid(weight_logits / temp)
        
        # === Subject Enhancement + Background Suppression ===
        # Map (0,1) to (suppression, enhancement)
        # Background region (attn near 0) -> multiply by suppress_factor (<1, fade)
        # Subject region (attn near 1) -> multiply by enhance_factor (>1, vivid)
        
        enhance_factor = F.softplus(self.enhance_scale) + 1.0  # >1.0
        suppress_factor = torch.sigmoid(self.suppress_scale)    # (0,1)
        
        # Hard threshold mapping: Nonlinear amplification of difference
        # Use power function to enhance contrast: High values higher, low values lower
        attn_enhanced = torch.pow(attn_base, 0.5)  # Subject region amplification
        
        # Combination: subject*enhancement + background*suppression
        # Formula: weight = suppress + (enhance - suppress) * attn_enhanced
        weight_map = suppress_factor + (enhance_factor - suppress_factor) * attn_enhanced
        
        # Additional contrast boost (learnable)
        contrast = torch.sigmoid(self.contrast_boost)
        weight_map = weight_map * (1.0 + contrast * 0.5)
        
        # Final weighting
        visual_output = x_resized * weight_map
        
        # Compensated feature compression
        feature_output = self.feature_compressor(features[-1])
        
        if original_size != (self.input_size, self.input_size):
            visual_output = F.interpolate(visual_output, size=original_size,
                                        mode='bilinear', align_corners=False)
        
        return visual_output, feature_output