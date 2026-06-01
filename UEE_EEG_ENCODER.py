#UEE
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (temporal + spatial)"""
    def __init__(self, C: int, T: int):
        super().__init__()
        
        # Temporal sinusoidal encoding [1, 1, T]
        temporal_pe = torch.zeros(1, 1, T)
        position_t = torch.arange(0, T, dtype=torch.float32).unsqueeze(0)
        div_term_t = torch.exp(torch.arange(0, 64, 2).float() * (-math.log(10000.0) / 64))
        
        pe_t = torch.zeros(1, T, 64)
        pe_t[0, :, 0::2] = torch.sin(position_t.T * div_term_t)
        pe_t[0, :, 1::2] = torch.cos(position_t.T * div_term_t)
        # Project to 1D
        temporal_pe[0, 0, :] = pe_t[0, :, :T].mean(dim=-1) if T <= 64 else pe_t[0, :, :T].mean(dim=-1)[:T]
        # Recompute to ensure dimension matching
        temporal_pe = torch.zeros(1, 1, T)
        position_t = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term_t = torch.exp(torch.arange(0, T, 2, dtype=torch.float32) * (-math.log(10000.0) / T))
        temporal_pe[0, 0, 0::2] = torch.sin(position_t[0::2, 0] * div_term_t[: (T + 1) // 2])
        if T > 1:
            temporal_pe[0, 0, 1::2] = torch.cos(position_t[1::2, 0] * div_term_t[: T // 2])
        
        # Spatial sinusoidal encoding [1, C, 1] - based on channel index
        spatial_pe = torch.zeros(1, C, 1)
        position_c = torch.arange(0, C, dtype=torch.float32).unsqueeze(1)
        div_term_c = torch.exp(torch.arange(0, C, 2, dtype=torch.float32) * (-math.log(10000.0) / C))
        spatial_pe[0, 0::2, 0] = torch.sin(position_c[0::2, 0] * div_term_c[: (C + 1) // 2])
        if C > 1:
            spatial_pe[0, 1::2, 0] = torch.cos(position_c[1::2, 0] * div_term_c[: C // 2])
        
        self.register_buffer('temporal_encoding', temporal_pe)
        self.register_buffer('spatial_encoding', spatial_pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.temporal_encoding + self.spatial_encoding


class AdaptiveInstanceNorm(nn.Module):
    """Instance normalization + global statistics aggregation"""
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta


class SingleHeadAttention(nn.Module):
    """Single-head attention"""
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = embed_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.norm(x)
        B, T, C = x.shape
        
        qkv = self.qkv_proj(x).reshape(B, T, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).reshape(B, T, C)
        out = self.out_proj(out)
        out = self.norm(out + x)
        
        return out.transpose(1, 2)


class SEBlock(nn.Module):
    """Channel attention"""
    def __init__(self, channels: int, reduction: int = 8, dropout: float = 0.1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class PIES(nn.Module):
    """
    Minimalist spatiotemporal adaptive filtering (always enabled, no info output)
    Enhance effective channels/time segments, equally suppress ineffective parts
    """
    def __init__(self, channels, reduction=4, temporal_kernel=7):
        super().__init__()
        
        # Channel gating (global importance)
        self.ch_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
        # Temporal gating (local temporal importance)
        self.time_gate = nn.Sequential(
            nn.Conv1d(channels, channels, temporal_kernel, 
                     padding=temporal_kernel//2, groups=channels),
            nn.BatchNorm1d(channels),
            nn.Sigmoid()
        )
        
        # Channel-temporal coupling
        self.coupling = nn.Sequential(
            nn.Conv1d(channels, channels//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Learnable residual ratio
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        B, C, T = x.shape
        
        # Channel gating [B, C, 1]
        ch_weight = self.ch_gate(x).unsqueeze(-1)
        
        # Temporal gating [B, C, T]
        time_weight = self.time_gate(x)
        
        # Joint gating
        joint_gate = ch_weight * time_weight
        coupling_factor = self.coupling(joint_gate.mean(dim=-1, keepdim=True))
        gate = joint_gate * coupling_factor
        gate = torch.clamp(gate, 0.01, 0.99)
        
        # Equal-amplitude modulation + residual connection
        purified = x * gate
        output = purified + self.residual_alpha * x
        
        return output


class UnifiedEEGEnhancer(nn.Module):
    """
    Unified enhancement module (integrated PIES purification, always enabled)
    Pipeline: PIES purification -> statistics extraction -> multi-scale convolution -> channel attention -> temporal attention -> adaptive modulation
    Interface fully compatible with original version: input [B,C,T], output [B,C,T]
    """
    def __init__(self, C: int, T: int, emb_dim: int = 8, kernel: int = 7, 
                 dropout: float = 0.1):
        super().__init__()
        self.C, self.T = C, T
        
        # PIES purification module (always enabled)
        self.pies = PIES(channels=C, reduction=8, temporal_kernel=7)
        
        # Statistics projection (based on purified signal)
        self.stats_proj = nn.Sequential(
            nn.Linear(2 * C, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale convolution
        self.ms_conv = nn.Sequential(
            nn.Conv1d(C, C, 1, bias=False),
            nn.Conv1d(C, C, kernel, padding=kernel//2, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(C, C // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(C // 8, C),
            nn.Sigmoid()
        )
        
        # Temporal attention
        self.temporal_norm = nn.LayerNorm(C)
        self.temporal_qkv = nn.Linear(C, C * 3)
        self.temporal_out = nn.Linear(C, C)
        self.temporal_scale = C ** -0.5
        
        # Adaptive modulation generation
        self.modulation_gen = nn.Sequential(
            nn.Linear(emb_dim, C),
            nn.ReLU(inplace=True),
            nn.Linear(C, C * 2)
        )
        
        self.out_conv = nn.Conv1d(C, C, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layerscale = nn.Parameter(torch.ones(1, C, 1) * 1e-4)
        
    def _temporal_attention(self, x):
        """Lightweight temporal self-attention"""
        x = x.transpose(1, 2)
        x = self.temporal_norm(x)
        B, T, C = x.shape
        
        qkv = self.temporal_qkv(x).reshape(B, T, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.temporal_scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).reshape(B, T, C)
        out = self.temporal_out(out)
        return out.transpose(1, 2)
        
    def _extract_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Extract signal statistical features"""
        mu = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        stats = torch.cat([mu, std], dim=1)
        stats = (stats - stats.mean(dim=-1, keepdim=True)) / (stats.std(dim=-1, keepdim=True) + 1e-8)
        return stats
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation (interface fully consistent with original version)
        
        Args:
            x: Input EEG [B, C, T]
            
        Returns:
            Enhanced features [B, C, T]
        """
        # Step 1: PIES purification (always executed)
        x = self.pies(x)
        
        # Step 2: Statistical feature extraction
        stats = self._extract_stats(x)
        emb = self.stats_proj(stats)
        
        # Step 3: Multi-scale convolution
        h = self.ms_conv(x)
        
        # Step 4: Channel attention
        ch_weight = self.channel_attn(h).unsqueeze(-1)
        h = h * ch_weight
        
        # Step 5: Temporal attention
        h_attn = self._temporal_attention(h)
        h = h + h_attn
        
        # Step 6: Adaptive modulation
        mod_params = self.modulation_gen(emb)
        channel_scale, gate = mod_params.chunk(2, dim=1)
        channel_scale = F.softplus(channel_scale).unsqueeze(-1) + 0.5
        gate = torch.sigmoid(gate).unsqueeze(-1)
        
        # Modulate and output
        h = self.out_conv(h * channel_scale * gate)
        h = self.dropout(h)
        
        # Final residual connection
        output = x + self.layerscale * h
        
        return output

class MultiResolutionFusion(nn.Module):
    """Multi-resolution fusion"""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.high_res_branch = nn.Identity()
        self.mid_res_branch = nn.Sequential(
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(embed_dim, embed_dim, 1, bias=False)
        )
        self.low_res_branch = nn.Sequential(
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.Conv1d(embed_dim, embed_dim, 1, bias=False)
        )
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.branch_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(3)
        ])
        
        self.res_weights = nn.Parameter(torch.ones(1, 3, 1))
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        
    def forward(self, x):
        B, D, T = x.shape
        
        if T < 4:
            return x
            
        feat_high = self.high_res_branch(x)
        feat_mid = self.mid_res_branch(x)
        feat_low = self.low_res_branch(x)

        feat_high = self.branch_norms[0](feat_high.transpose(1,2)).transpose(1,2)
        feat_mid = self.branch_norms[1](feat_mid.transpose(1,2)).transpose(1,2)
        feat_low = self.branch_norms[2](feat_low.transpose(1,2)).transpose(1,2)
        
        target_size = x.shape[-1]
        feat_mid_up = F.interpolate(feat_mid, size=target_size, mode='linear', align_corners=False)
        feat_low_up = F.interpolate(feat_low, size=target_size, mode='linear', align_corners=False)
        
        all_feats = torch.stack([feat_high, feat_mid_up, feat_low_up], dim=1)
        all_feats = all_feats.permute(0, 3, 1, 2).reshape(B * target_size, 3, D)
        
        q = self.q_proj(all_feats).view(B * target_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(all_feats).view(B * target_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(all_feats).view(B * target_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B * target_size, 3, D)
        out = self.out_proj(out)
        
        weights = F.softmax(self.res_weights, dim=1)
        out = (out * weights).sum(dim=1)
        
        return out.reshape(B, target_size, D).transpose(1, 2)


class ImprovedFrequencyConv(nn.Module):
    """Improved frequency convolution"""
    def __init__(self, in_channels, out_channels, fs=250, dropout=0.1, freq_bands=None):
        super().__init__()
        
        if freq_bands is None:
            freq_bands = {'delta': (1, 4), 'theta': (4, 8), 
                         'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}
        
        self.num_bands = len(freq_bands)
        mid_ch = ((out_channels // self.num_bands + 3) // 4) * 4
        total_mid = mid_ch * self.num_bands
        
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, total_mid, 1, bias=False),
            AdaptiveInstanceNorm(total_mid),
            nn.ReLU(inplace=True)
        )
        
        self.processors = nn.ModuleList()
        for f_low, f_high in freq_bands.values():
            avg_f = (f_low + f_high) / 2
            kernel = max(5, min(25, int(fs / avg_f / 2)))
            if kernel % 2 == 0: 
                kernel += 1
            
            self.processors.append(nn.Sequential(
                nn.Conv1d(mid_ch, mid_ch, kernel, padding=kernel//2),
                AdaptiveInstanceNorm(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_ch, mid_ch, 1),
                AdaptiveInstanceNorm(mid_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ))
        
        self.cross_q = nn.ModuleList([nn.Linear(mid_ch, mid_ch) for _ in range(self.num_bands)])
        self.cross_k = nn.ModuleList([nn.Linear(mid_ch, mid_ch) for _ in range(self.num_bands)])
        self.cross_v = nn.ModuleList([nn.Linear(mid_ch, mid_ch) for _ in range(self.num_bands)])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(mid_ch) for _ in range(self.num_bands)])
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(total_mid, total_mid//8),
            nn.ReLU(inplace=True),
            nn.Linear(total_mid//8, self.num_bands),
            nn.Sigmoid()
        )
        
        self.output = nn.Sequential(
            nn.Conv1d(total_mid, out_channels, 1),
            AdaptiveInstanceNorm(out_channels),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        B, C, T = x.shape
        res = self.residual(x)
        
        x = self.input_proj(x)
        chunks = torch.chunk(x, self.num_bands, dim=1)
        
        feats = [proc(chk) for proc, chk in zip(self.processors, chunks)]
        global_feats = [f.mean(-1) for f in feats]
        global_feats = [F.normalize(f, p=2, dim=-1) for f in global_feats]
        
        enhanced = []
        for i in range(self.num_bands):
            q = self.cross_q[i](global_feats[i]).unsqueeze(1)
            others = torch.stack([global_feats[j] for j in range(self.num_bands) if j != i], dim=1)
            k, v = self.cross_k[i](others), self.cross_v[i](others)
            
            attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1]**0.5), dim=-1)
            attended = torch.matmul(attn, v).squeeze(1)
            enhanced.append(feats[i] * torch.sigmoid(attended.unsqueeze(-1)))
        
        stacked = torch.cat(enhanced, dim=1)
        weights = self.gate(stacked).view(B, self.num_bands, 1, 1)
        
        stacked = stacked.view(B, self.num_bands, -1, T) * weights
        return self.output(stacked.view(B, -1, T)) + res


class TimeDomainMultiScaleConv(nn.Module):
    """Time-domain multi-scale convolution"""
    def __init__(self, in_channels: int, out_channels: int, 
                 base_stride: int = 25, dropout: float = 0.1):
        super().__init__()
        mid_channels = out_channels // 2
        
        self.pyramid_downsample = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=50, stride=50,
                     padding=25, bias=False),
            AdaptiveInstanceNorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.multi_res_fusion = MultiResolutionFusion(mid_channels)
        
        self.direct_path = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels // 4, kernel_size=1, bias=False),
            AdaptiveInstanceNorm(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.aspp_convs = nn.ModuleList()
        dilations = [1, 3, 5, 7, 9]
        aspp_channels = out_channels // 4
        
        for d in dilations:
            self.aspp_convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        mid_channels, mid_channels,
                        kernel_size=25, stride=1,
                        padding=12*d, dilation=d,
                        bias=False
                    ),
                    AdaptiveInstanceNorm(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(mid_channels, aspp_channels, 1, bias=False),
                    AdaptiveInstanceNorm(aspp_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                )
            )
        
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(mid_channels, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        total_channels = out_channels // 4 + len(dilations)*aspp_channels + out_channels // 4
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(total_channels, total_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(total_channels // 8, total_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.fusion_conv = nn.Conv1d(total_channels, out_channels, 1, bias=False)
        self.fusion_norm = AdaptiveInstanceNorm(out_channels)
        
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=out_channels, num_heads=4, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        
        pyramid_feat = self.pyramid_downsample(x)
        target_len = pyramid_feat.shape[-1]
        pyramid_feat = self.multi_res_fusion(pyramid_feat)
        
        aspp_feats = [conv(pyramid_feat) for conv in self.aspp_convs]
        direct_feat = self.direct_path(pyramid_feat)
        global_feat = self.global_branch(pyramid_feat)
        
        direct_feat = F.interpolate(direct_feat, size=target_len, mode='linear', align_corners=False)
        global_feat = F.interpolate(global_feat, size=target_len, mode='linear', align_corners=False)
        
        all_features = [direct_feat] + aspp_feats + [global_feat]
        fused = torch.cat(all_features, dim=1)
        gate = self.fusion_gate(fused)
        fused = fused * gate
        
        fused = self.fusion_conv(fused)
        fused = self.fusion_norm(fused)
        fused = F.relu(fused, inplace=True).transpose(1, 2)

        fused = self.attn_norm(fused)
        attn_out, _ = self.temporal_attn(fused, fused, fused)
        attn_out = self.attn_norm(attn_out + fused)
        
        return attn_out


class EEGEncoder(nn.Module):
    """Cross-subject stable EEG encoder"""
    def __init__(self, C: int, T: int, embed_dim: int, out_dim: int,
                 stride: int = 25, num_layers: int = 1,
                 dropout: float = 0.1, fs: int = 250,
                 freq_bands=None):
        super().__init__()
        self.C, self.T = C, T
        self.embed_dim = embed_dim
        
        self.pos_encoding = PositionalEncoding(C, T)
        self.input_norm = AdaptiveInstanceNorm(C)
        
        self.enhancers = nn.ModuleList([
            UnifiedEEGEnhancer(C, T, emb_dim=8, kernel=7, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.freq_conv = ImprovedFrequencyConv(
            in_channels=C, 
            out_channels=embed_dim,
            fs=fs, 
            dropout=dropout,
            freq_bands=freq_bands
        )
        
        self.time_conv = TimeDomainMultiScaleConv(
            in_channels=C,
            out_channels=embed_dim,
            base_stride=stride,
            dropout=dropout
        )
        
        self.fusion_weights = nn.Parameter(torch.zeros(2))
        self.fusion_temp = nn.Parameter(torch.ones(1) * 0.5)
        
        self.attention_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1)
        )
        
        self.projection = nn.Linear(embed_dim, out_dim, bias=True)
        self.output_norm = nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        
        x = self.input_norm(x)
        x = self.pos_encoding(x)

        #UEE
        for enhancer in self.enhancers:
            x = enhancer(x)
        
        freq_features = self.freq_conv(x)
        time_features = self.time_conv(x)
        time_features = time_features.transpose(1, 2)
        
        if freq_features.shape[-1] != time_features.shape[-1]:
            min_len = min(freq_features.shape[-1], time_features.shape[-1])
            freq_features = F.interpolate(freq_features, size=min_len, mode='linear', align_corners=False)
            time_features = F.interpolate(time_features, size=min_len, mode='linear', align_corners=False)
        
        fusion_weights = F.softmax(self.fusion_weights / (self.fusion_temp.abs() + 0.1), dim=0)

        freq_features = F.layer_norm(freq_features, freq_features.shape[-1:])
        time_features = F.layer_norm(time_features, time_features.shape[-1:])
        
        combined_features = (fusion_weights[0] * freq_features + fusion_weights[1] * time_features)
        
        combined_features = combined_features.transpose(1, 2)
        
        attention_scores = self.attention_scorer(combined_features)
        attention_weights = F.softmax(attention_scores, dim=1)
        fused_feature = (combined_features * attention_weights).sum(dim=1)
        
        output = self.projection(fused_feature)
        return self.output_norm(output)