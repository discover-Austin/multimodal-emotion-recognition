import torch
import torch.nn as nn

class ModalityFusion(nn.Module):
    def __init__(self, image_feature_dim: int, text_feature_dim: int, num_emotions: int = 7):
        super(ModalityFusion, self).__init__()
        fusion_dim = image_feature_dim + text_feature_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, num_emotions)
        )

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):
        fused = torch.cat([image_features, text_features], dim=1)
        logits = self.fusion_layer(fused)
        return logits