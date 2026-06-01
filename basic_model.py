import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


from UEE_EEG_ENCODER import EEGEncoder
from data_augmentation import EEGAugmentation, ImageAugmentation, FeatureSpaceAugmentation
from PPA import EEGToImageAttention
from SAVE import DilatedSubjectWeighting


class RobustProjectionHead(nn.Module):
    """High-robustness projection head"""
    def __init__(self, in_dim: int, out_dim: int, modality: str, 
                 depth: int = 3, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.modality = modality
        self.depth = depth
        hidden_dim = in_dim * expansion
        
        if modality == 'eeg':
            self.dropout = dropout
        else:
            self.dropout = dropout * 0.5
        
        layers = []
        layers.extend([
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout)
        ])
        
        for i in range(depth - 2):
            layers.append(
                ResidualBlock(
                    hidden_dim,
                    dropout=self.dropout,
                    use_bn=(modality == 'eeg')
                )
            )
        
        layers.extend([
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim)
        ])
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_norm(x)
        return self.net(x)


class ResidualBlock(nn.Module):
    """Pre-activation residual block"""
    def __init__(self, dim: int, dropout: float, 
                 use_bn: bool = True, expand_factor: float = 0.5):
        super().__init__()
        hidden_dim = int(dim * expand_factor)
        
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, dim, bias=False)
        )
    
    def forward(self, x):
        return x + self.block(x)


class EmotionAligner(nn.Module):
    def __init__(self, C: int = 63, T: int = 250, H: int = 500, W: int = 500,
                 img_split: int = 3, device: str = 'cuda',
                 use_robust_proj: bool = True,
                 proj_expansion: int = 4,
                 proj_depth: int = 3,
                 num_codewords: int = 512, #512
                 topk: int = 128, #64
                 codebook_layers: int = 3,
                 codebook_expansion: int = 5):
        super().__init__()
        
        self.eeg_channel = C
        self.eeg_time = T
        self.img_split = img_split
        self.device = device
        self.H = H
        self.W = W
        self.dropout = 0.3 #0.3

        self.embed_dim = 512
        self.out_dim = 512

        # Data augmentation
        self.probability = 0.1
        self.eeg_aug = EEGAugmentation(
            probability=self.probability,
            C=self.eeg_channel,
            T=self.eeg_time
        )
        self.img_aug = ImageAugmentation(probability=self.probability)
        self.eeg_feat_aug = FeatureSpaceAugmentation(
            feature_dim=self.out_dim,
            noise_std=0.01,
            scale_range=(0.95, 1.05),
            dropout_prob=0.05,
            prob=self.probability
        )
        self.img_feat_aug = FeatureSpaceAugmentation(
            feature_dim=self.out_dim,
            noise_std=0.02,
            scale_range=(0.9, 1.1),
            dropout_prob=0.05,
            prob=self.probability
        )
        
        # EEG encoder
        self.encode_eeg = EEGEncoder(
            C=self.eeg_channel,
            T=self.eeg_time,
            embed_dim=self.embed_dim,
            out_dim=self.out_dim,
            dropout=self.dropout
        )

        # Image subject extractor
        self.img_subject_extractor = DilatedSubjectWeighting(feature_dim=self.out_dim)
        self.weight_img_feat = nn.Parameter(torch.zeros(2))
        
        # Image encoder (CLIP)
        self.clip_model_name = "!!!Your Path!!!" + "/clip_model/ViT-B-32.pt"
        self.clip_input_size = 224
        
        self.clip_model, self.preprocess = clip.load(
            self.clip_model_name, 
            device=device,
            jit=False
        )
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.float()
        self.encode_image = self.clip_model.visual
        self.resize_image = self._resize_with_interpolation


        # Unidirectional EEG->Image attention (pure feature transformation)
        self.eeg_to_img_attention = EEGToImageAttention(
            dim=self.out_dim,
            num_layers=codebook_layers,
            num_codewords=num_codewords,
            topk=topk,
            dropout=self.dropout,
            expansion=codebook_expansion
        )
        
        # Projection heads
        self.res_here = nn.Parameter(torch.tensor(1.0))
        if use_robust_proj:
            self.eeg_projection = RobustProjectionHead(
                in_dim=self.out_dim,
                out_dim=self.out_dim,
                modality='eeg',
                depth=5,
                expansion=3,
                dropout=self.dropout
            )
            self.img_projection = RobustProjectionHead(
                in_dim=self.out_dim,
                out_dim=self.out_dim,
                modality='img',
                depth=5,
                expansion=2,
                dropout=self.dropout / 2
            )
        else:
            self.eeg_projection = nn.Linear(self.out_dim, self.out_dim)
            self.img_projection = nn.Linear(self.out_dim, self.out_dim)
    
    def _resize_with_interpolation(self, img_tensor: torch.Tensor, 
                                   target_size: tuple = (224, 224)) -> torch.Tensor:
        if img_tensor.shape[-2:] != target_size:
            img_tensor = F.interpolate(
                img_tensor, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
        return img_tensor
    
    def forward_projection(self, feat: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == 'eeg':
            return self.eeg_projection(feat)
        else:
            return self.img_projection(feat)
    
    def forward(self, eeg: torch.Tensor, img: torch.Tensor):

        # Data augmentation (training only)
        if self.training:
            eeg = self.eeg_aug(eeg)
            img = self.img_aug(img)
        
        # EEG encoding + projection
        eeg_feat = self.encode_eeg(eeg)
        eeg_feat = self.forward_projection(eeg_feat, modality='eeg') * self.res_here + eeg_feat
        
        # Image encoding + projection
        img_resized = self.resize_image(img, target_size=(224, 224))
        # Image subject extraction with background weakening
        img_resized, img_weight_feat = self.img_subject_extractor(img_resized)
        with torch.no_grad():
            img_resized = img_resized.to(torch.float32)
            with torch.cuda.amp.autocast(enabled=False):
                img_feat = self.encode_image(img_resized)
        img_feat = self.forward_projection(img_feat, modality='img') * self.res_here + img_feat

        img_feat_weight = torch.softmax(self.weight_img_feat, dim=0)
        img_feat = img_feat_weight[0] * F.normalize(img_feat, p=2, dim=-1) + img_feat_weight[1] * F.normalize(img_weight_feat, p=2, dim=-1)

        # Core: Unidirectional attention (pure feature transformation)
        # Pass img_feat during training for assistance, None during testing
        eeg_enhanced = self.eeg_to_img_attention(eeg_feat, img_feat if self.training else None)

        # Feature space augmentation (training only)
        if self.training:
            eeg_enhanced = self.eeg_feat_aug(eeg_enhanced)
            img_feat = self.img_feat_aug(img_feat)
        
        # Final normalization
        eeg_feat = F.normalize(eeg_enhanced, p=2, dim=-1)
        img_feat = F.normalize(img_feat, p=2, dim=-1)
        
        return eeg_feat, img_feat