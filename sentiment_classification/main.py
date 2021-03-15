import json
import pickle

import torch
from fastapi import FastAPI
from torchtext import data

from sentiment_classification.model import TextClassificationModel

MODEL = './sentiment_classification/model_path/sentiment_model.pth'
MODEL_CONF = './sentiment_classification/model_path/metadata.json'
VOCAB = './sentiment_classification/model_path/vocab.pk'
app = FastAPI()

with open(MODEL_CONF, 'r') as fp:
    config = json.load(fp)

with open(VOCAB, 'rb') as fp:
    vocab = pickle.load(fp)

model = TextClassificationModel(
    config['vocab_size'],
    config['embedding_size'],
    config['label_size'],
    predict=True
)
model.load_state_dict(torch.load(MODEL))
tokenizer = data.utils.get_tokenizer('basic_english')


def text_pipeline(x):
    return [vocab[token] for token in tokenizer(x)]


@app.get('/classify/{text}')
def get_prediction(text: str):
    processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
    offsets = torch.tensor([0])
    with torch.no_grad():
        prediction = model(processed_text, offsets).flatten()
    return {
        "prediction": prediction.argmax().item(),
        "confidence": prediction.max().item()
    }
