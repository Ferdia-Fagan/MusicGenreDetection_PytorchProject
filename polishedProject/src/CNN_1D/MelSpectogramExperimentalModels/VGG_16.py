import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Basic standard VGG_16 implementation as a base for 1D.
    """
    def __init__(self, params):
        super(Net, self).__init__()


        # Block 1:
        self.conv_1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Block 2:
        self.conv_3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.conv_4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        #block 3
        self.conv_5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.conv_6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.conv_7 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        #block 4
        self.conv_8 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.conv_9 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.conv_10 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        #block 5
        self.conv_11 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.conv_12 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.conv_13 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )



        self.fcblock1 = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fcblock2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.activation = nn.Sequential(
            nn.Linear(1024, 10),
        )

        self.apply(self._init_weights)

    def forward(self, input):
        """
        Input: (batch_size, time windows, mel_filters)
        """
        input = input.transpose(1, 2)
        # (batch_size, time_length, mel_filters, time_length)

        #block 1    128x304
        conv_out = self.conv_1(input)
        conv_out = self.conv_2(conv_out)
        # block 2   64x150
        conv_out = self.conv_3(conv_out)
        conv_out = self.conv_4(conv_out)
        # block 3   128x73
        conv_out = self.conv_5(conv_out)
        conv_out = self.conv_6(conv_out)
        conv_out = self.conv_7(conv_out)
        # block 4   256x33
        conv_out = self.conv_8(conv_out)
        conv_out = self.conv_9(conv_out)
        conv_out = self.conv_10(conv_out)
        # block 5   512x13
        conv_out = self.conv_11(conv_out)
        conv_out = self.conv_12(conv_out)
        conv_out = self.conv_13(conv_out)
        # 512x3

        conv_out = conv_out.view(input.shape[0], conv_out.size(1) * conv_out.size(2))

        fc_out = self.fcblock1(conv_out)
        fc_out = self.fcblock2(fc_out)
        logit = self.activation(fc_out)

        return logit


    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)





















