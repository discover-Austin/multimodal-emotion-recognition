import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms
from src.data.preprocessing.image_preprocessing import preprocess_image
from src.data.preprocessing.text_preprocessing import preprocess_text

class MultiModalDataset(Dataset):
    def __init__(self, jsonl_file: str, tokenizer: BertTokenizer, image_transform=None):
        self.samples = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Load and process image
        image = Image.open(sample['image_path']).convert("RGB")
        image = preprocess_image(image, transform=self.image_transform)
        # Process text
        encoded_text = preprocess_text(sample['text'], tokenizer=self.tokenizer)
        label = int(sample['label'])
        return {
            "image": image,
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def create_dataloader(jsonl_file: str, tokenizer, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4):
    dataset = MultiModalDataset(jsonl_file, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)