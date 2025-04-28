import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss for Super Resolution
    Adapted from the original ArcFace paper for image super-resolution tasks.
    
    Args:
        feature_dim (int): Dimension of the feature embeddings
        s (float): Scaling factor for the cosine values
        m (float): Margin parameter to enforce separation between classes
        easy_margin (bool): Use the easy margin version
    """
    def __init__(self, feature_dim=32, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.feature_dim = feature_dim
        self.s = s  # scale factor
        self.m = m  # margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        # Feature extractor for both LR and HR images
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, sr_tensor, hr_tensor):
        """
        Forward pass for ArcFace loss
        
        Args:
            sr_tensor (torch.Tensor): Super-resolved images [B, C, H, W]
            hr_tensor (torch.Tensor): High-resolution ground truth images [B, C, H, W]
            
        Returns:
            torch.Tensor: Computed ArcFace loss
        """
        # Extract features from both SR and HR images
        sr_features = self.feature_extractor(sr_tensor)  # [B, feature_dim]
        hr_features = self.feature_extractor(hr_tensor)  # [B, feature_dim]
        
        # Normalize features to be unit vectors
        sr_features = F.normalize(sr_features, p=2, dim=1)
        hr_features = F.normalize(hr_features, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.cosine_similarity(sr_features, hr_features, dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
        
        # Compute arcface angles with margin
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Convert to distance loss (1 - phi) and scale
        arcface_loss = (1.0 - phi) * self.s
        
        return arcface_loss.mean()


class FeatureExtractorArcFace(nn.Module):
    """
    A feature extractor module to be used with the main SR model
    Extracts features from both SR and HR images for ArcFace loss computation
    """
    def __init__(self, in_channels=3, feature_dim=64):
        super(FeatureExtractorArcFace, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, x):
        return self.features(x)
