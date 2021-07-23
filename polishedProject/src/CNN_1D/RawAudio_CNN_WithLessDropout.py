import torch
import torch.nn as nn


class Net(nn.Module):
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
            # nn.Conv1d(128, 256, kernel_size=6, stride=5, padding=1),
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
            nn.Dropout(0.25),
        )

        self.fcblock1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.fcblock2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.fcblock3 = nn.Sequential(
            nn.Linear(512, 10),
        )

        self.apply(self._init_weights)

    def forward(self, x):
        x = x.transpose(1, 2)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)

        out = out.view(out.size(0), -1)

        logit = self.fcblock2(out)
        logit = self.fcblock3(logit)

        return logit

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
