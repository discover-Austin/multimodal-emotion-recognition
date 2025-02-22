import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class EmotionVisionTransformer(nn.Module):
    def __init__(self, num_emotions: int = 7, pretrained: bool = True):
        super(EmotionVisionTransformer, self).__init__()
        config = ViTConfig()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224") if pretrained else ViTModel(config)
        hidden_size = self.vit.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_emotions)
        )

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.vit(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits