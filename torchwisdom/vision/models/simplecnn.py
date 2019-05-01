import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_chan=3, num_classes=3, input_size=(224,224), use_sigmoid=False):
        super(SimpleCNN, self).__init__()
        self.input_size = input_size
        self.in_chan = in_chan
        self.feature = nn.Sequential(
            nn.Conv2d(in_chan, 32, kernel_size=3),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, kernel_size=3),
            nn.Conv2d(8, 8, kernel_size=3),
        )

        classifier_feature = self._classifier_feature_num()
        if use_sigmoid:
            self.classifer = nn.Sequential(
                nn.Linear(classifier_feature, num_classes),
                nn.Sigmoid()
            )
        self.classifer = nn.Linear(classifier_feature, num_classes)

    def _classifier_feature_num(self):
        x = torch.rand(1, self.in_chan, self.input_size[0], self.input_size[1])
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x.size()[1]

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = SimpleCNN(3, 3)
    out = model(x)
    print(out.shape)