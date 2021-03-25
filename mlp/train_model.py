import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mlp.model import MLP, MLPDataset

MODEL_PATH = 'model_path/mlp.pth'
data_set = MLPDataset('data/data.csv')
train_set, val_set, test_set = random_split(data_set, lengths=(6000, 2000, 2000))

train_loader = DataLoader(train_set, batch_size=8)
val_loader = DataLoader(val_set, batch_size=8)
test_loader = DataLoader(test_set, batch_size=8)

model = MLP()
optimizer_fn = optimizer.SGD(params=model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    writer = SummaryWriter('./log')
    for ep in range(epoch):
        train_loss, t_idx = 0, 1
        train_acc, train_total = 0, 0
        with tqdm(train_loader, unit="batch") as t_epoch:
            t_epoch.set_description(f"Epoch: {ep}")
            for t_idx, (label, inp) in enumerate(t_epoch, start=1):
                optimizer_fn.zero_grad()
                output = model(inp)
                loss = criterion(output, label)
                loss.backward()
                optimizer_fn.step()
                train_acc += torch.sum((output.argmax(1) == label).to(int)).item()
                train_total += label.size(0)
                train_loss += loss.item()
                t_epoch.set_postfix(train_loss=train_loss / t_idx, train_acc=train_acc / train_total)

        val_loss, idx = 0, 1
        val_acc, val_total = 0, 0
        with tqdm(val_loader, unit="batch") as v_epoch:
            t_epoch.set_description(f"Epoch: {ep}")
            for idx, (label, inp) in enumerate(v_epoch, start=1):
                with torch.no_grad():
                    output = model(inp)
                    val_acc += torch.sum((output.argmax(1) == label).to(int)).item()
                    val_total += label.size(0)
                    loss = criterion(output, label)
                    val_loss += loss.item()
                v_epoch.set_postfix(val_loss=val_loss / idx, val_acc=val_acc / val_total)

        writer.add_scalars('Accuracy', {
            'Training': train_acc / train_total,
            'Validation': val_acc / val_total
        }, ep)
        writer.add_scalars('Loss', {
            'Training': train_loss / t_idx,
            'Validation': val_loss / idx
        }, ep)
    writer.close()

    test_loss, idx = 0, 1
    for idx, (label, inp) in enumerate(test_loader):
        with torch.no_grad():
            output = model(inp)
            loss = criterion(output, label)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / idx}")

    if epoch > 30:
        torch.save(model.state_dict(), MODEL_PATH)


if __name__ == '__main__':
    train(100)
