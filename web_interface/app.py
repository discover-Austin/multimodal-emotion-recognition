import io
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms

from src.models.emotion_vision_transformer import EmotionVisionTransformer
from src.models.emotion_bert import EmotionBERT
from src.models.modality_fusion import ModalityFusion

app = FastAPI(
    title="MultiModal Emotion Recognition API",
    description="Real-time emotion recognition from webcam feed and text sentiment analysis using transformer-based models.",
    version="1.0"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_emotions = 7

# Initialize models
vision_model = EmotionVisionTransformer(num_emotions=num_emotions, pretrained=True).to(device)
text_model = EmotionBERT(num_emotions=num_emotions, pretrained=True).to(device)
image_feature_dim = vision_model.vit.config.hidden_size
text_feature_dim = text_model.bert.config.hidden_size
fusion_model = ModalityFusion(image_feature_dim=image_feature_dim, text_feature_dim=text_feature_dim, num_emotions=num_emotions).to(device)

vision_model.eval()
text_model.eval()
fusion_model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

emotion_labels = {
    0: "happy",
    1: "sad",
    2: "angry",
    3: "fear",
    4: "surprised",
    5: "disgust",
    6: "neutral"
}

@app.post("/predict", summary="Predict Emotion from Image and Text")
async def predict_emotion(text: str = Form(...), image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_image = image_transform(pil_image).unsqueeze(0).to(device)
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoded_text["input_ids"].to(device)
        attention_mask = encoded_text["attention_mask"].to(device)
    
        with torch.no_grad():
            image_features = vision_model.vit(processed_image).last_hidden_state[:, 0, :]
            text_features = text_model.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            fusion_logits = fusion_model(image_features, text_features)
            predicted_index = torch.argmax(fusion_logits, dim=1).item()
            confidence = torch.softmax(fusion_logits, dim=1)[0, predicted_index].item()
    
        result = {
            "predicted_emotion": emotion_labels.get(predicted_index, "unknown"),
            "emotion_index": predicted_index,
            "confidence": confidence
        }
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("web_interface.app:app", host="0.0.0.0", port=8000, reload=False)