import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Handles 2D mel spectograms.
    Basically the same architecture as 1D,
    except contains a MaxPool2d at each convolution block
    (instead of previosly were I did not use pool at layer 1 or 4).
    """
    def __init__(self,params):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(params.L1_conv_input_filters_n, 64,
                      kernel_size=params.L1_conv_kernel_size,
                      stride=params.L1_conv_stride,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=params.L1_maxPool2d_kernel_size)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=params.L2_conv_kernel_size,
                      stride=params.L2_conv_stride,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=params.L2_maxPool2d_kernel_size)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=params.L3_conv_kernel_size,
                      stride=params.L3_conv_stride,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=params.L3_maxPool2d_kernel_size)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, params.L5_conv_output, kernel_size=params.L4_conv_kernel_size,
                      stride=params.L4_conv_stride,padding=1),
            nn.BatchNorm2d(params.L5_conv_output),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=params.L4_maxPool2d_kernel_size),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(params.L5_conv_output, params.L5_conv_output, kernel_size=params.L5_conv_kernel_size,
                      stride=params.L5_conv_stride,padding=1),
            nn.BatchNorm2d(params.L5_conv_output),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=params.L5_maxPool2d_kernel_size),
        )

        # self.globalAveragePooling = nn.Sequential(
        #     nn.Conv2d(512, 10, kernel_size=1,
        #               stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d((1,8)),
        #     nn.Flatten()
        # )

        self.globalAveragePooling1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1,
                      stride=1),
            nn.Dropout(0.25),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.AvgPool1d(16),    # if you make this MaxPool1d() instead, will not get good results. This si sbecause avg is linear, were as max is non-linear
            # nn.Flatten()
        )

        self.globalAveragePooling2 = nn.Sequential(
            nn.Conv2d(512, 10, kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1,8)),    # if you make this MaxPool1d() instead, will not get good results. This si sbecause avg is linear, were as max is non-linear
            nn.Flatten()
        )

        # self.fcblock1 = nn.Sequential(
        #     nn.Linear(params.fullyConnected_B1_input_n, params.fullyConnected_B1_input_n),
        #     nn.Dropout(0.5),
        #     nn.ReLU()
        # )
        #
        # self.fcblock2 = nn.Sequential(
        #     nn.Linear(params.fullyConnected_B1_input_n, 512),
        #     nn.Dropout(0.5),
        #     nn.ReLU()
        # )
        #
        # self.activation = nn.Sequential(
        #     nn.Linear(512, 10),
        #     # nn.LogSoftmax(dim=1)
        #     # because I am using "crossEntropyLoss" I will be remove LogSoftmax
        #
        # )

        self.apply(self._init_weights)

    def forward(self, input):
        """
        Input: (batch_size, time windows, mel_filters)
        """
        input = torch.unsqueeze(input, 1)
        # (batch_size, 1, time windows, mel_filters)
        conv_out = self.conv1(input)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = self.conv4(conv_out)
        conv_out = self.conv5(conv_out)
        # [batch_size, 512, 4, 1]
        # dense = conv_out.view(input.size(0), -1)
        # [batch_size, 2048]
        # fc_out = self.fcblock1(dense)
        # fc_out = self.fcblock2(fc_out)
        # logit = self.activation(fc_out)

        globalPooling_out = self.globalAveragePooling1(conv_out)
        logit = self.globalAveragePooling2(globalPooling_out)

        return logit

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)




