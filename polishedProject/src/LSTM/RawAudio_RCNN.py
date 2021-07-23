import torch
import torch.nn as nn


class Net(nn.Module):
    """
    input: raw waveform (1D)
    input filters through multiple convolution layers,
    to then filter through LSTM block.
    """
    def __init__(self, params):
        super(Net, self).__init__()

        """
        kernel_size and stride will be chosen at value 50% or 100% 
        (the reason for this is because these are comparable to raw 
        mel spectogram models, as they use a fourier transform with 
        parameters window_size and hop_size of this ratio. )
        """
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3,
                      stride=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3,
                      stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=2,
                      stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=4,
                      stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=4,
                      stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

        self.lstm = nn.LSTM(512, 256, 2,
                             batch_first=True)


        self.fcblock1 = nn.Sequential(
            nn.Linear(5632, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fcblock2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.activation = nn.Sequential(
            nn.Linear(1024, 10),
        )

        self.apply(self._init_weights)

    def forward(self, input):
        """
        Input: (batch_size, time_length, amplitude domain (its not divided like mel, so its 1D))
        """

        input = input.transpose(1, 2)
        # Input: (batch_size, amplitude domain (1D), time_length)

        # Convolution layers taking input of the mel spectrogram along
        # the time domain: [batch_size, 1, 61740]
        conv_out = self.conv1(input)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = self.conv4(conv_out)
        conv_out = self.conv5(conv_out)
        conv_out = self.conv6(conv_out)
        conv_out = self.conv7(conv_out)
        # out: [batch_size, channels(512), time segments(22)]

        # After convolution layers I then have to flatten the 2D convolutions
        # this is technically two operations.
        conv_out = conv_out.transpose(1,2)
        # conv_out = conv_out.contiguous().view(conv_out.size()[0], conv_out.size()[2],-1)
        # (batch_size, time sequence (22), channels (512))
        lstm_out, _ = self.lstm(conv_out)
        # (batch_size, 22, 256)

        dense = lstm_out.contiguous().view(lstm_out.size()[0], -1)
        # [batch_size, 5632]
        fc_out = self.fcblock1(dense)
        fc_out = self.fcblock2(fc_out)
        logit = self.activation(fc_out)

        return logit

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
