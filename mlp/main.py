
import torch
from fastapi import FastAPI

from mlp.model import MLP


MODEL_PATH = './mlp/model_path/mlp.pth'
app = FastAPI()
model = MLP(inference=True)
model.load_state_dict(torch.load(MODEL_PATH))


@app.get('/classify')
def get_classification(
    feature_1: int,
    feature_2: int,
    feature_3: int,
    feature_4: int,
    feature_5: int,
    feature_6: int,
    feature_7: int,
    feature_8: int,
    feature_9: int,
    feature_10: int,
):
    inp = torch.tensor([
        feature_1,
        feature_2,
        feature_3,
        feature_4,
        feature_5,
        feature_6,
        feature_7,
        feature_8,
        feature_9,
        feature_10
    ], dtype=torch.float32)
    with torch.no_grad():
        output = model(inp)
        print(output)
    return {
        'class': output.argmax().item(),
        'confidence': output.max().item()
    }
