import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Tested 1D raw mel spectograms using this simple CNN I built
    inspired by AlexNet and VGG_16. Just trying to get a taste here.
    """
    def __init__(self, params):
        super(Net, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3,
                      stride=1),
            nn.ReLU(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )

        self.maxPoolingLayer1 = nn.MaxPool1d(kernel_size=2)

        self.conv_3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3,
                      stride=1),
            nn.ReLU(),
        )

        self.conv_4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3,
                      stride=1),
            nn.ReLU(),
        )

        self.conv_5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2)
        )


        self.fcblock1 = nn.Sequential(
            nn.Linear(9216, 4096),      # far to many parameters for my FC layers.
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.fcblock2 = nn.Sequential(
            nn.Linear(4096, 1000),
            nn.ReLU(),
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

        conv_out = self.conv_1(input)
        conv_out = self.conv_2(conv_out)
        conv_out = self.maxPoolingLayer1(conv_out)

        conv_out = self.conv_3(conv_out)
        conv_out = self.conv_4(conv_out)
        conv_out = self.conv_5(conv_out)

        dense = conv_out.view(input.shape[0], conv_out.size(1) * conv_out.size(2))

        fc_out = self.fcblock1(dense)
        fc_out = self.fcblock2(fc_out)
        logit = self.activation(fc_out)

        return logit


    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
        # elif isinstance(layer, nn.ReLU):
        #     nn.init.kaiming_uniform_(layer.weight)





















