import json
from collections import Counter
import pickle

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext import data, datasets
from torchtext.vocab import Vocab

from sentiment_classification.model import TextClassificationModel

SEED = 0
PATH = './model_path/sentiment_model.pth'
model_metadata = './model_path/metadata.json'
vocab_data = './model_path/vocab.pk'

torch.manual_seed(SEED)

tokenizer = data.utils.get_tokenizer('basic_english')
train_iter, test_iter = datasets.AG_NEWS(split=('train', 'test'))
counter = Counter()
label_set = set()
for (label_, line) in train_iter:
    counter.update(tokenizer(line))
    label_set.add(label_)
vocab = Vocab(counter, min_freq=1)
vocab_size = len(vocab)
EMBEDDING_SIZE = 64
label_size = len(label_set)

with open(model_metadata, 'w') as fp:
    json.dump({
        'embedding_size': EMBEDDING_SIZE,
        'label_size': label_size,
        'vocab_size': vocab_size
    }, fp)

with open(vocab_data, 'wb') as fp:
    pickle.dump(vocab, fp)


def text_pipeline(x):
    return [vocab[token] for token in tokenizer(x)]


def label_pipeline(x):
    return int(x) - 1


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets


train_iter, test_iter = datasets.AG_NEWS(split=('train', 'test'))
train_data, test_data = list(train_iter), list(test_iter)
train_loader = DataLoader(train_data, batch_size=8, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=8, collate_fn=collate_batch)

model = TextClassificationModel(vocab_size, EMBEDDING_SIZE, label_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.01)


def train(epochs):
    print("Start train")
    for ep in range(1, epochs + 1):
        epoch_loss = 0
        correct = 0
        total_count = 0
        for label, text, offsets in train_loader:
            label: torch.Tensor
            optimizer.zero_grad()
            predicted: torch.Tensor = model(text, offsets)
            loss = criterion(predicted, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            correct += (predicted.argmax(1) == label).sum().item()
            total_count += label.size(0)
        print(f"Epoch: {ep}; Loss: {epoch_loss / total_count}, Accuracy: {correct / total_count}")
    torch.save(model.state_dict(), PATH)
    print("Training done")


def test():
    with torch.no_grad():
        test_loss = 0
        test_correct = 0
        total_count = 0
        for label, text, offsets in test_loader:
            label: torch.Tensor
            predicted = model(text, offsets)
            loss = criterion(predicted, label)
            test_loss += loss.item()
            test_correct += (predicted.argmax(1) == label).sum().item()
            total_count += label.size(0)
        print(f"Test; Loss: {test_loss / total_count}, Accuracy: {test_correct / total_count}")


if __name__ == '__main__':
    train(10)
    test()
