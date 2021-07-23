import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Basic alexnet implementation as a base for 1D
    """
    def __init__(self, params):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=11,
                      stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5,
                      stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.fcblock1 = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fcblock2 = nn.Sequential(
            nn.Linear(1280, 1000),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.activation = nn.Sequential(
            nn.Linear(1000, 10),
        )

        self.apply(self._init_weights)

    def forward(self, input):
        """
        Input: (batch_size, time windows, mel_filters)
        """

        input = input.transpose(1, 2)
         # (batch_size, mel_filters, time_length)
        # 128 x 304
        conv_out = self.layer1(input)
        conv_out = self.layer2(conv_out)
        conv_out = self.layer3(conv_out)
        # 256 x 5
        dense = conv_out.view(input.shape[0], conv_out.size(1) * conv_out.size(2))
        # [kernel_size, 1280]
        fc_out = self.fcblock1(dense)
        fc_out = self.fcblock2(fc_out)

        logit = self.activation(fc_out)

        return logit


    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)





















