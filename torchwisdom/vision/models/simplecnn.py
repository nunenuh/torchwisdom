import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_chan=3, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_chan, 32, kernel_size=3),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, kernel_size=3),
        )

        self.classifer = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)
    model = SimpleCNN(3, 3)
    out = model(x)
    print(out.shape)