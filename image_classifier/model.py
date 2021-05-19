import torch.nn as nn
import torch.nn.functional as f

import torchvision.transforms as t


transforms = t.Compose([
    t.ToTensor(),
    t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class ImageModel(nn.Module):
    def __init__(self, predict=False):
        super(ImageModel, self).__init__()
        self.predict = predict
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(f.relu(self.conv1(x)))
        x = self.pool1(f.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        if self.predict:
            return f.softmax(x, dim=1)
        return x
