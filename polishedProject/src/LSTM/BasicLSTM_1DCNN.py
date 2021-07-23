import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Input: 1D raw mel frequency filters/channels.
    CNN filters by mel frequency filters /channels along the time domain.
    This was to experiment with the opposite of the other
    RCNN pattern I followed. Ie:
    [LSTM * 2]->[conv->BN->ReLU]->[conv->BN->ReLU->pooling]*2->[conv->BN->ReLU->conv(strided)->BN->ReLU]
    """
    def __init__(self,params):
        super(Net, self).__init__()

        self.lstm = nn.LSTM(params.input_dim, params.hidden_dim, params.lstm_n_layers,
                             batch_first=True)

        self.conv_l1 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2)
        )

        self.conv_l2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )

        self.conv_l3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
        )

        self.conv_l4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3,
                      stride=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3),
        )

        self.fcblock1 = nn.Sequential(
            nn.Linear(2816, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fcblock2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.activation = nn.Sequential(
            nn.Linear(512, 10),
        )

        self.apply(self._init_weights)


    def forward(self, input):
        """
        Input: (batch_size, time_length, mel_filters)
        """

        lstm_out, _ = self.lstm(input)

        # (batch_size, time_segments, lstm hidden channels)

        lstm_out = lstm_out.transpose(1,2)

        # (batch_size, lstm hidden channels, time_segments)
        # [batch_size, 256 304
        conv_out = self.conv_l1(lstm_out)
        conv_out = self.conv_l2(conv_out)
        conv_out = self.conv_l3(conv_out)
        conv_out = self.conv_l4(conv_out)
        #out: [batch_size, (256) filters/channels, 11]

        dense = conv_out.contiguous().view(conv_out.size()[0], -1)
        # [batch_size, 2816]
        fc_out = self.fcblock1(dense)
        # [batch_size, 1024]
        fc_out = self.fcblock2(fc_out)
        # [batch_size, 512]
        logit = self.activation(fc_out)
        return logit

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
















