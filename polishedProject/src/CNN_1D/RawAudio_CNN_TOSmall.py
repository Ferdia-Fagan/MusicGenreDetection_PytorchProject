import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self,params):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            # nn.Conv1d(1, 128, kernel_size=6, stride=5, padding=0),
            nn.Conv1d(1, 128, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2,stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=3)
        )
        self.conv4 = nn.Sequential(
            # nn.Conv1d(128, 256, kernel_size=6, stride=5, padding=1),
            nn.Conv1d(128, 256, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=3)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4,stride=4)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4,stride=4)
        )


        self.fcblock2 = nn.Sequential(
            nn.Linear(13312, 512),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        self.fcblock3 = nn.Sequential(
            nn.Linear(512, 10),
        )

        self.apply(self._init_weights)

    def forward(self, x):

        x = x.transpose(1, 2)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        out = out.view(x.shape[0], out.size(1) * out.size(2))
        logit = self.fcblock2(out)
        logit = self.fcblock3(logit)

        return logit

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
