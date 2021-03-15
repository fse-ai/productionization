import torch
import torch.nn as nn
import torch.nn.functional as f


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_class, predict=False):
        super(TextClassificationModel, self).__init__()
        self.predict = predict
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_size, sparse=True)
        self.fc = nn.Linear(embedding_size, num_class)
        self.embedding.weight.data.uniform_(-0.5, 0.5)
        self.fc.weight.data.uniform_(-0.5, 0.5)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        if self.predict:
            return f.softmax(self.fc(embedded))
        return self.fc(embedded)
