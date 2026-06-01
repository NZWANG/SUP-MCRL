import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class EEGAugmentation(nn.Module):
    """EEG data augmentation module - Pure PyTorch implementation"""
    def __init__(self, probability: float = 0.5, C: int = 63, T: int = 250):
        super().__init__()
        self.probability = probability
        self.C = C
        self.T = T
        
    def time_jitter(self, eeg: torch.Tensor) -> torch.Tensor:
        """Random temporal shift (zero-padded)"""
        if torch.rand(1).item() > self.probability:
            return eeg
        
        shift = torch.randint(-self.T // 10, self.T // 10 + 1, (1,)).item()
        
        if shift == 0:
            return eeg
        
        # eeg shape: [C, T]
        if shift > 0:
            # Shift right: pad left with zeros, truncate right
            return F.pad(eeg[:, :-shift], (shift, 0), value=0)
        else:
            # Shift left: pad right with zeros, truncate left
            shift_abs = -shift
            return F.pad(eeg[:, shift_abs:], (0, shift_abs), value=0)
    
    def amplitude_scaling(self, eeg: torch.Tensor) -> torch.Tensor:
        """Amplitude scaling"""
        if torch.rand(1).item() > self.probability:
            return eeg
        
        scale = torch.empty(1).uniform_(0.8, 1.2).item()
        return eeg * scale
    
    def channel_dropout(self, eeg: torch.Tensor) -> torch.Tensor:
        """Random channel dropout (set to zero)"""
        if torch.rand(1).item() > self.probability:
            return eeg
        
        dropout_prob = 0.1
        mask = torch.bernoulli(torch.full((self.C, 1), 1 - dropout_prob, device=eeg.device))
        return eeg * mask
    
    def channel_shuffle(self, eeg: torch.Tensor) -> torch.Tensor:
        """Random channel shuffling"""
        if torch.rand(1).item() > self.probability:
            return eeg
        
        indices = torch.randperm(self.C, device=eeg.device)
        return eeg[indices, :]
    
    def add_gaussian_noise(self, eeg: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise"""
        if torch.rand(1).item() > self.probability:
            return eeg
        
        noise_std = 0.01 * eeg.std()
        noise = torch.randn_like(eeg) * noise_std
        return eeg + noise
    
    def frequency_mask(self, eeg: torch.Tensor) -> torch.Tensor:
        """Frequency mask (simulate band filtering)"""
        if torch.rand(1).item() > self.probability:
            return eeg
        
        mask_width = torch.randint(10, 31, (1,)).item()  # [10, 30]
        start_idx = torch.randint(0, max(1, self.T - mask_width), (1,)).item()
        
        eeg = eeg.clone()
        eeg[:, start_idx:start_idx + mask_width] = 0
        return eeg
    
    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        eeg: [B, C, T]
        """
        B = eeg.shape[0]
        augmented = []
        
        for i in range(B):
            x = eeg[i]
            
            # Apply series of augmentations
            x = self.time_jitter(x)
            x = self.amplitude_scaling(x)
            x = self.channel_dropout(x)
            x = self.channel_shuffle(x)
            x = self.add_gaussian_noise(x)
            x = self.frequency_mask(x)
            
            augmented.append(x.unsqueeze(0))
        
        return torch.cat(augmented, dim=0)


class ImageAugmentation(nn.Module):
    """Image data augmentation module - Pure PyTorch implementation"""
    def __init__(self, probability: float = 0.5):
        super().__init__()
        self.probability = probability
        
    def random_flip(self, img: torch.Tensor) -> torch.Tensor:
        """Random horizontal flip"""
        if torch.rand(1).item() > self.probability:
            return img
        
        return torch.flip(img, dims=[-1])
    
    def random_rotation(self, img: torch.Tensor) -> torch.Tensor:
        """Batch random rotation (±10 degrees) - More efficient"""
        B = img.shape[0]
        
        # Determine which samples need rotation
        mask = torch.rand(B, device=img.device) < self.probability
        
        if not mask.any():
            return img
        
        # Generate different angles for each sample
        angles = torch.empty(B, device=img.device).uniform_(-10, 10)
        angle_rad = angles * torch.pi / 180.0
        
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        
        # Create rotation matrices
        theta = torch.zeros(B, 2, 3, device=img.device, dtype=torch.float32)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        
        # Non-rotated samples use identity matrix
        identity = torch.eye(2, 3, device=img.device, dtype=torch.float32).unsqueeze(0)
        theta = torch.where(mask.unsqueeze(-1).unsqueeze(-1), theta, identity)
        
        grid = F.affine_grid(theta, img.size(), align_corners=False)
        return F.grid_sample(img, grid, align_corners=False)
    
    def color_jitter(self, img: torch.Tensor) -> torch.Tensor:
        """Color jitter (brightness, contrast)"""
        if torch.rand(1).item() > self.probability:
            return img
        
        # Brightness adjustment
        brightness_factor = torch.empty(1).uniform_(0.8, 1.2).item()
        img = img * brightness_factor
        
        # Contrast adjustment
        contrast_factor = torch.empty(1).uniform_(0.8, 1.2).item()
        mean = img.mean(dim=[-2, -1], keepdim=True)
        img = (img - mean) * contrast_factor + mean
        
        return torch.clamp(img, 0, 1)
    
    def random_crop_resize(self, img: torch.Tensor) -> torch.Tensor:
        """Random crop and resize back to original size"""
        if torch.rand(1).item() > self.probability:
            return img
        
        B, C, H, W = img.shape
        crop_scale = torch.empty(1).uniform_(0.8, 1.0).item()
        new_H, new_W = int(H * crop_scale), int(W * crop_scale)
        
        # Random crop position
        top = torch.randint(0, max(1, H - new_H), (1,)).item()
        left = torch.randint(0, max(1, W - new_W), (1,)).item()
        
        # Crop
        img_cropped = img[:, :, top:top + new_H, left:left + new_W]
        
        # Resize back to original size
        img_resized = F.interpolate(
            img_cropped, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        return img_resized
    
    def gaussian_blur(self, img: torch.Tensor) -> torch.Tensor:
        """Gaussian blur"""
        if torch.rand(1).item() > self.probability:
            return img
        
        kernel_size = 3
        sigma = torch.empty(1).uniform_(0.1, 2.0).item()
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=img.device)
        coords -= (kernel_size - 1) / 2.0
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        
        kernel_2d = kernel[:, None] * kernel[None, :]
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size)
        
        # Apply blur
        padding = kernel_size // 2
        return F.conv2d(img, kernel_2d, padding=padding, groups=3)
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [B, 3, H, W], range [0, 1]
        """
        B = img.shape[0]
        augmented = []
        
        for i in range(B):
            x = img[i:i+1]  # Keep batch dimension
            
            # Apply series of augmentations
            x = self.random_flip(x)
            x = self.random_rotation(x)
            x = self.color_jitter(x)
            x = self.random_crop_resize(x)
            x = self.gaussian_blur(x)
            
            augmented.append(x)
        
        return torch.cat(augmented, dim=0)

class FeatureSpaceAugmentation(nn.Module):
    """
    Feature space augmentation module after projection head
    
    Purpose: Apply controllable perturbations on L2-normalized feature vectors to enhance contrastive learning robustness
    Advantage: Does not modify original EEG/image encodings, only adjusts distribution in contrastive space
    """
    
    def __init__(
        self, 
        feature_dim: int,
        noise_std: float = 0.01,          # Gaussian noise standard deviation
        scale_range: tuple = (0.9, 1.1),  # Random scaling range
        dropout_prob: float = 0.1,        # Feature dropout probability
        prob: float = 0.5                 # Overall application probability
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.dropout_prob = dropout_prob
        self.prob = prob
        
        # Learnable noise intensity (adaptive adjustment)
        self.log_noise_scale = nn.Parameter(torch.log(torch.tensor(noise_std)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Projection head output, shape (batch, feature_dim)
        Output: Augmented features, shape (batch, feature_dim), re-normalized with L2
        """
        if not self.training or torch.rand(1).item() > self.prob:
            return x  # Evaluation mode or return original features directly
        
        batch_size = x.size(0)
        
        # 1. Random scaling (independent per sample)
        # Generate [batch, 1] scaling factors
        scale = torch.rand(batch_size, 1, device=x.device) 
        scale = scale * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        
        # 2. Adaptive Gaussian noise
        # Noise intensity dynamically adjusts during training, limited range to prevent excessive values
        noise_std = torch.exp(self.log_noise_scale).clamp(0.001, 0.05)
        
        # Generate [batch, feature_dim] noise
        noise = torch.randn_like(x) * noise_std
        
        # 3. Feature dropout (randomly zero out partial dimensions)
        if self.dropout_prob > 0:
            dropout_mask = torch.rand_like(x) > self.dropout_prob
            x = x * dropout_mask.float()
        
        # 4. Apply augmentation
        # x is already normalized features, scale first then add noise
        x_aug = x * scale + noise
        
        # 5. Re-apply L2 normalization (critical! Maintain unit sphere distribution)
        x_aug = F.normalize(x_aug, p=2, dim=-1)
        
        return x_aug