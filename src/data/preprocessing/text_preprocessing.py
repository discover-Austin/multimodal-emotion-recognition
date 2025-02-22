from transformers import BertTokenizer

def preprocess_text(text: str, tokenizer: BertTokenizer, max_length: int = 128):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )