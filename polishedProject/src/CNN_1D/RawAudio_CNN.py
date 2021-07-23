import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Takes in input of raw waveform sampled at 22050 in ~3
    second segments.
    (I would add input transformation from DataProcessing
    to resample but I dont have time)
    """
    def __init__(self,params):
        super(Net, self).__init__()

        """
        kernel_size and stride will be chosen at value 50% or 100% 
        (the reason for this is because these are comparable to raw 
        mel spectogram models, as they use a fourier transform with 
        parameters window_size and hop_size of this ratio. )
        """
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=params.L1_conv_kernel_size,
                      stride=params.L1_conv_kernel_stride),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=params.L2_conv_kernel_size,
                      stride=params.L2_conv_kernel_stride,padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=params.L2_avgPool_kernel_size)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=params.L3_conv_kernel_size,
                      stride=params.L3_conv_kernel_stride,padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=params.L3_avgPool_kernel_size)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=params.L4_conv_kernel_size,
                      stride=params.L4_conv_kernel_stride,padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=params.L4_avgPool_kernel_size)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=params.L5_conv_kernel_size,
                      stride=params.L5_conv_kernel_stride,padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=params.L5_avgPool_kernel_size)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=params.L6_conv_kernel_size,
                      stride=params.L6_conv_kernel_stride,padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=params.L6_avgPool_kernel_size)
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=params.L7_conv_kernel_size,
                      stride=params.L7_conv_kernel_stride,padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=params.L7_avgPool_kernel_size)
        )

        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=params.L8_conv_kernel_size,
                      stride=params.L8_conv_kernel_stride,padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=params.L8_avgPool_kernel_size)
        )

        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=params.L9_conv_kernel_size,
                      stride=params.L9_conv_kernel_stride,padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=params.L9_avgPool_kernel_size)
        )

        self.conv10 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=params.L10_conv_kernel_size,
                      stride=params.L10_conv_kernel_stride,padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.fcblock1 = nn.Sequential(
            nn.Linear(params.fc1_input, params.fc1_output),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fcblock2 = nn.Sequential(
            nn.Linear(params.fc1_output, params.fc2_output),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.activation = nn.Sequential(
            nn.Linear(params.fc2_output, 10),
        )

        self.apply(self._init_weights)

    def forward(self, input):
        """
        Input: (batch_size, time_length, amplitude (1D))
        """

        input = input.transpose(1, 2)
        # [batch_size, 1, 61740]

        conv_out = self.conv1(input)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = self.conv4(conv_out)
        conv_out = self.conv5(conv_out)
        conv_out = self.conv6(conv_out)
        conv_out = self.conv7(conv_out)
        conv_out = self.conv8(conv_out)
        conv_out = self.conv9(conv_out)
        conv_out = self.conv10(conv_out)
        # [batch_size, 512, 2]
        dense = conv_out.view(conv_out.size(0), -1)
        # [batch_size, 1024]
        fc_out = self.fcblock1(dense)
        fc_out = self.fcblock2(fc_out)

        logit = self.activation(fc_out)

        return logit

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
