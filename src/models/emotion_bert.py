import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class EmotionBERT(nn.Module):
    def __init__(self, num_emotions: int = 7, pretrained: bool = True):
        super(EmotionBERT, self).__init__()
        config = BertConfig()
        self.bert = BertModel.from_pretrained("bert-base-uncased") if pretrained else BertModel(config)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_emotions)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits