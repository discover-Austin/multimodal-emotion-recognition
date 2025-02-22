import os
import argparse
import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from src.models.emotion_vision_transformer import EmotionVisionTransformer
from src.models.emotion_bert import EmotionBERT
from src.models.modality_fusion import ModalityFusion
from src.utils.data_loader import create_dataloader

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Logging started. Log file: {log_file}")

def train_epoch(vision_model, text_model, fusion_model, dataloader, criterion, optimizer, device):
    vision_model.train()
    text_model.train()
    fusion_model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Image modality
        image_outputs = vision_model.vit(images)
        image_features = image_outputs.last_hidden_state[:, 0, :]
        image_logits = vision_model.classifier(image_features)

        # Text modality
        text_outputs = text_model.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output
        text_logits = text_model.classifier(text_features)

        # Fusion modality
        fusion_logits = fusion_model(image_features, text_features)

        # Compute losses
        loss_image = criterion(image_logits, labels)
        loss_text = criterion(text_logits, labels)
        loss_fusion = criterion(fusion_logits, labels)
        loss = (loss_image + loss_text + loss_fusion) / 3.0

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = fusion_logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples if total_samples else 0.0
    accuracy = total_correct / total_samples if total_samples else 0.0
    return epoch_loss, accuracy

def validate_epoch(vision_model, text_model, fusion_model, dataloader, criterion, device):
    vision_model.eval()
    text_model.eval()
    fusion_model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            image_outputs = vision_model.vit(images)
            image_features = image_outputs.last_hidden_state[:, 0, :]
            image_logits = vision_model.classifier(image_features)

            text_outputs = text_model.bert(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.pooler_output
            text_logits = text_model.classifier(text_features)

            fusion_logits = fusion_model(image_features, text_features)

            loss_image = criterion(image_logits, labels)
            loss_text = criterion(text_logits, labels)
            loss_fusion = criterion(fusion_logits, labels)
            loss = (loss_image + loss_text + loss_fusion) / 3.0

            running_loss += loss.item() * images.size(0)
            preds = fusion_logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    epoch_loss = running_loss / total_samples if total_samples else 0.0
    accuracy = total_correct / total_samples if total_samples else 0.0
    return epoch_loss, accuracy

def save_checkpoint(state, checkpoint_dir, filename):
    os.makedirs(checkpoint_dir, exist_ok=True)
    file_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, file_path)
    logging.info(f"Checkpoint saved: {file_path}")

def main(args):
    setup_logging(args.log_dir)
    logging.info("Training process started.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataloader = create_dataloader(args.train_file, tokenizer, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = create_dataloader(args.val_file, tokenizer, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logging.info("Data loaders initialized.")

    num_emotions = args.num_emotions
    vision_model = EmotionVisionTransformer(num_emotions=num_emotions, pretrained=True).to(device)
    text_model = EmotionBERT(num_emotions=num_emotions, pretrained=True).to(device)
    image_feature_dim = vision_model.vit.config.hidden_size
    text_feature_dim = text_model.bert.config.hidden_size
    fusion_model = ModalityFusion(image_feature_dim=image_feature_dim, text_feature_dim=text_feature_dim, num_emotions=num_emotions).to(device)
    logging.info("Models initialized.")

    criterion = nn.CrossEntropyLoss()
    all_parameters = list(vision_model.parameters()) + list(text_model.parameters()) + list(fusion_model.parameters())
    optimizer = optim.AdamW(all_parameters, lr=args.learning_rate)

    best_val_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        logging.info(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(vision_model, text_model, fusion_model, train_dataloader, criterion, optimizer, device)
        logging.info(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")

        val_loss, val_acc = validate_epoch(vision_model, text_model, fusion_model, val_dataloader, criterion, device)
        logging.info(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            state = {
                "epoch": epoch,
                "vision_model_state": vision_model.state_dict(),
                "text_model_state": text_model.state_dict(),
                "fusion_model_state": fusion_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "args": vars(args)
            }
            checkpoint_filename = f"best_checkpoint_epoch_{epoch}.pth"
            save_checkpoint(state, args.checkpoint_dir, checkpoint_filename)

    logging.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MultiModal Emotion Recognition System")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training JSONL file")
    parser.add_argument("--val_file", type=str, required=True, help="Path to the validation JSONL file")
    parser.add_argument("--num_emotions", type=int, default=7, help="Number of emotion classes")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for saving logs")
    args = parser.parse_args()
    main(args)