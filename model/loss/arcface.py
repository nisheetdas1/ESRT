import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss for Super Resolution
    Adapted for use with ESRT model's internal features.
    
    Args:
        s (float): Scaling factor for the cosine values
        m (float): Margin parameter to enforce separation between classes
        easy_margin (bool): Use the easy margin version
    """
    def __init__(self, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.s = s  # scale factor
        self.m = m  # margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        # Global pooling and normalization
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, sr_features, hr_features):
        """
        Forward pass for ArcFace loss
        
        Args:
            sr_features (torch.Tensor): Features from super-resolved images 
            hr_features (torch.Tensor): Features from high-resolution images
            
        Returns:
            torch.Tensor: Computed ArcFace loss
        """
        # Apply global average pooling and flatten
        sr_features = self.gap(sr_features).view(sr_features.size(0), -1)
        hr_features = self.gap(hr_features).view(hr_features.size(0), -1)
        
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


class ExtractFeaturesForArcFace(nn.Module):
    """
    A hook-like module that extracts intermediate features from ESRT model for ArcFace loss
    """
    def __init__(self):
        super(ExtractFeaturesForArcFace, self).__init__()
    
    def forward(self, model, lr_tensor, hr_tensor):
        """
        Extract features from both LR and SR paths for ArcFace calculation
        
        Args:
            model: The ESRT model
            lr_tensor: Low-resolution input tensor
            hr_tensor: High-resolution ground truth tensor
            
        Returns:
            Tuple of (sr_features, hr_features, sr_output)
        """
        # Extract features from the input HR image
        with torch.no_grad():
            # Extract HR features at the same layer depth as we will for SR
            hr_features = model.head(hr_tensor)
            for i in range(model.n_blocks):
                hr_features = model.body[i](hr_features)
        
        # Forward pass through the model to get SR features and output
        lr_features = model.head(lr_tensor)
        # Capture intermediate features before final upsampling
        sr_features = lr_features
        for i in range(model.n_blocks):
            sr_features = model.body[i](sr_features)
            
        # Continue to generate the SR output for standard L1 loss
        body_out = [sr_features]
        res1 = torch.cat(body_out, 1)
        res1 = model.reduce(res1)
        sr_output = model.tail(res1)
        sr_output = model.up(lr_features) + sr_output
        
        return sr_features, hr_features, sr_output
