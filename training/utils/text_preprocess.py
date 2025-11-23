import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d{10,11}', '', text)
    text = re.sub(r'[^\w\s\.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
