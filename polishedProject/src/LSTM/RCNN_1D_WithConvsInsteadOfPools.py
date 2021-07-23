import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Input: 1D raw mel frequency filters/channels.
    CNN filters by mel frequency filters /channels along the time domain.
    This followed RCNN pattern described in the paper. With structure:
    [convolution layer]*4 -> [LSTM*2]
    The difference with RCNN_1D is that I used convolution layers with stride
    to replace maxpooling.
    """
    def __init__(self,params):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=params.conv_l1_kernel_size,
                      stride=params.conv_l1_stride, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=params.conv_l2_kernel_size,
                      stride=params.conv_l2_stride, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=params.l2_maxPool_kernel_size,
                      stride=params.l2_maxPool_kernel_size, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=params.conv_l3_kernel_size,
                      stride=params.conv_l3_stride, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=params.l3_maxPool_kernel_size,
                      stride=params.l3_maxPool_kernel_size, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=params.conv_l4_kernel_size,
                      stride=params.conv_l4_stride, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=2,
                      stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(params.l5_LSTM_input_dim, params.l5_LSTM_hidden_dim,
                            params.l5_LSTM_n_layers, batch_first=True)

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
        input = input.transpose(1, 2)
        # [batch_size, 64, 128]
        conv_out = self.conv1(input)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = self.conv4(conv_out)
        # [batch_size, 256, 18]
        conv_out = conv_out.transpose(1,2)
        # conv_out = conv_out.contiguous().view(conv_out.size()[0], conv_out.size()[2],-1)
        lstm_out , _ = self.lstm(conv_out)
        # [batch_size, 18, 256,]
        dense = lstm_out.contiguous().view(lstm_out.size()[0], -1)
        # [batch_size, 4608]
        fc_out = self.fcblock1(dense)
        # [batch_size, 1024]
        fc_out = self.fcblock2(fc_out)
        # [batch_size, 512]

        logits = self.activation(fc_out)

        return logits

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

