import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        """
        kernel_size and stride will be chosen at value 50% or 100% 
        (the reason for this is because these are comparable to raw 
        mel spectogram models, as they use a fourier transform with 
        parameters window_size and hop_size of this ratio. )
        """
        self.conv1 = nn.Sequential(
            # nn.Conv1d(1, 128, kernel_size=6, stride=5, padding=0),
            nn.Conv1d(1, 128, kernel_size=3,
                      stride=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 19683 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=4,
                      stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        # 6561 x 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=4,
                      stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        # 2187 x 128
        self.conv4 = nn.Sequential(
            # nn.Conv1d(128, 256, kernel_size=6, stride=5, padding=1),
            nn.Conv1d(128, 128, kernel_size=4,
                      stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        # 729 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4,
                      stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

        self.conv6 = nn.Sequential(
            # nn.Conv1d(128, 256, kernel_size=6, stride=5, padding=1),
            nn.Conv1d(256, 512, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 729 x 256
        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=2,
                      stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fcblock1 = nn.Sequential(
            nn.Linear(9216, 512),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        self.fcblock2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        self.fcblock3 = nn.Sequential(
            nn.Linear(256, 10),
            # nn.Softmax(dim=1)
            # because I am using "crossEntropyLoss" I will be remove LogSoftmax
        )

        # self.fc = nn.Linear(256, 10)
        #
        # self.activation = nn.Softmax(dim=1)

        self.apply(self._init_weights)

    def forward(self, x):

        x = x.transpose(1, 2)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)

        # out = out.view(x.shape[0], out.size(1) * out.size(2))
        out = out.view(out.size(0), -1)

        logit = self.fcblock1(out)
        logit = self.fcblock2(logit)
        logit = self.fcblock3(logit)

        return logit

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
