import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Input: 2D raw mel spectogram.
    Followed RCNN pattern described in paper.
    [LSTM*2]->[conv->BN->ReLU->pooling]*3
    """
    def __init__(self,params):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=params.conv_l1_kernel_size,
                      stride=params.conv_l1_stride, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=params.l1_maxPool_kernel_size)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=params.conv_l2_kernel_size,
                      stride=params.conv_l2_stride, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=params.l2_maxPool_kernel_size)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=params.conv_l3_kernel_size,
                      stride=params.conv_l3_stride, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=params.l3_maxPool_kernel_size)
        )

        self.input_dim = params.l5_LSTM_input_dim
        self.hidden_dim = params.l5_LSTM_hidden_dim
        self.n_layers = params.l5_LSTM_n_layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers,
                             batch_first=True)

        self.fcblock1 = nn.Sequential(
            nn.Linear(params.l6_fullyConnected1_input_n, 1024),
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

        input = torch.unsqueeze(input, 1)
        # (batch_size, 1, time_length, mel_filters)
        # (batch_size, 1, 304, 128)
        conv_out = self.conv1(input)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        # (batch_size, 256, 16, 6)
        conv_out = conv_out.transpose(1,2)
        conv_out = conv_out.contiguous().view(conv_out.size()[0], conv_out.size()[1],-1)
        # had to use contiquous above because of transpose. I believe these two lines
        # are equivelent to bottom, but I am not sure.
        # conv_out = conv_out.contiguous().view(out.size()[0], out.size()[2], -1)

        # [batch_size, 16 (time segment), 1536 (channels)]
        lstm_out, _ = self.lstm(conv_out)
        # [batch_size, 16, 256]
        lstm_out = lstm_out.contiguous().view(conv_out.size()[0], -1)
        # [batch_size, 4096]
        out = self.fcblock1(lstm_out)
        out = self.fcblock2(out)
        out = self.activation(out)

        return out


    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)


