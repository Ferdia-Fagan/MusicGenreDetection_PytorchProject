import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Same architecture as
    MelSpectogram_CNN_MoreDropout.
    Replaced MaxPooling layers with strided [conv->BN->ReLU] blocks.
    I did did increase filters size for output to these blocks to compensate for loss in resolution due to striding.
    """
    def __init__(self,params):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(params.L1_conv_input_filters_n, 64, kernel_size=params.L1_conv_kernel_size,
                      stride=params.L1_conv_stride),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=params.L2_conv_kernel_size,
                      stride=params.L2_conv_stride),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=params.L2_maxPool2d_kernel_size,stride=params.L2_maxPool2d_kernel_size, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=params.L3_conv_kernel_size,
                      stride=params.L3_conv_stride),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128,kernel_size=params.L3_maxPool2d_kernel_size, stride=params.L3_maxPool2d_kernel_size, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=params.L4_conv_kernel_size,
                      stride=params.L4_conv_stride),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=params.L5_conv_kernel_size,
                      stride=params.L5_conv_stride),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=params.L5_maxPool2d_kernel_size, stride=params.L5_maxPool2d_kernel_size,
                      padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )

        self.fcblock1 = nn.Sequential(
            nn.Linear(params.fullyConnected_B1_input_n, params.fullyConnected_B1_out_n),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )

        self.fcblock2 = nn.Sequential(
            nn.Linear(params.fullyConnected_B1_out_n, params.fullyConnected_B2_out_n),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.activation = nn.Sequential(
            nn.Linear(512, 10),
        )

        self.apply(self._init_weights)

    def forward(self, input):
        """
        Input: (batch_size, time windows, mel_filters)
        """
        input = input.transpose(1, 2)
        # 128 x 152
        conv_out = self.conv1(input)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = self.conv4(conv_out)
        conv_out = self.conv5(conv_out)
        # 128 x 16
        dense = conv_out.view(input.shape[0], conv_out.size(1) * conv_out.size(2))
        # [batch_size, 128]
        fc_out = self.fcblock1(dense)
        fc_out = self.fcblock2(fc_out)
        logit = self.activation(fc_out)

        return logit


    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)





















