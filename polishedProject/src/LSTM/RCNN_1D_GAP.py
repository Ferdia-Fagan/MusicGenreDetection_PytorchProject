import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Input: 1D raw mel frequency filters/channels.
    CNN filters by mel frequency filters /channels along the time domain.
    This followed RCNN pattern described in the paper. With structure:
    [convolution layer]*4 -> [LSTM*2]
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
            nn.MaxPool1d(kernel_size=params.l2_maxPool_kernel_size),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=params.conv_l3_kernel_size,
                      stride=params.conv_l3_stride, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=params.l3_maxPool_kernel_size),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=params.conv_l4_kernel_size,
                      stride=params.conv_l4_stride, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=params.l4_maxPool_kernel_size),
        )

        self.lstm = nn.LSTM(params.l5_LSTM_input_dim, params.l5_LSTM_hidden_dim,
                            params.l5_LSTM_n_layers, batch_first=True)

        # self.globalAveragePooling = nn.Sequential(
        #     nn.Conv1d(256, 10, kernel_size=1,
        #               stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool1d(16),
        #     nn.Flatten()
        # )

        self.globalAveragePooling1 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1,
                      stride=1),
            nn.Dropout(0.25),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # nn.AvgPool1d(16),    # if you make this MaxPool1d() instead, will not get good results. This si sbecause avg is linear, were as max is non-linear
            # nn.Flatten()
        )

        self.globalAveragePooling2 = nn.Sequential(
            nn.Conv1d(128, 10, kernel_size=1,
                      stride=1),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(16),    # if you make this MaxPool1d() instead, will not get good results. This si sbecause avg is linear, were as max is non-linear
            nn.Flatten()
        )

        # self.fcblock1 = nn.Sequential(
        #     nn.Linear(params.l6_fullyConnected1_input_n, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        # )
        #
        # self.fcblock2 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        # )
        #
        # self.activation = nn.Sequential(
        #     nn.Linear(512, 10),
        # )

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
        # [batch_size, 256, 17]
        # out = out.contiguous().view(out.size()[0], out.size()[2],-1)
        conv_out = conv_out.transpose(1, 2)
        # [batch_size, 17, 256]
        lstm_out , _ = self.lstm(conv_out)

        globalAveragePooling_Output = self.globalAveragePooling1(lstm_out.transpose(1,2))
        logit = self.globalAveragePooling2(globalAveragePooling_Output)

        # dense = lstm_out.contiguous().view(lstm_out.size()[0], -1)
        #
        #
        # # [batch_size, 4352]
        # fc_out = self.fcblock1(dense)
        # fc_out = self.fcblock2(fc_out)
        #
        # logits = self.activation(fc_out)

        return logit

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

