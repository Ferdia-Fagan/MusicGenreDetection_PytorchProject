import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Input: 2D raw mel spectogram.
    similar pattern:
    [LSTM*2]->[conv->BN->ReLU->pooling]*3
    """
    def __init__(self,params):
        super(Net, self).__init__()

        self.lstm = nn.LSTM(params.input_dim, params.hidden_dim, params.lstm_n_layers,
                             batch_first=True)

        self.conv_l1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=4,
                      stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.conv_l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4,
                      stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.conv_l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4,
                      stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        """
         something to note here is the ratio here for the linear layer.
         if I had done 1536:1024 for fc1, and 1024:512 for fc2, 
         the accuracy goes way down. 
         This could be due to to many parameters, leading to a long training time
         (so maybe if I trained for longer it would be better).
        """

        self.fcblock1 = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fcblock2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.activation = nn.Sequential(
            nn.Linear(256, 10),
        )

        self.apply(self._init_weights)


    def forward(self, input):
        """
        Input: (batch_size, time_length, mel_filters)
        """

        lstm_out, _ = self.lstm(input)
        # [batch_size, [304] time_sequence, [128] hidden_layers]
        # unsqueezed = torch.unsqueeze(lstm_out, 1)
        unsqueezed = lstm_out.contiguous().view(lstm_out.size()[0], 1, lstm_out.size()[1], lstm_out.size()[2])
        # [batch_size, 1, time_sequence, hidden_layers]

        conv_out = self.conv_l1(unsqueezed)
        conv_out = self.conv_l2(conv_out)
        conv_out = self.conv_l3(conv_out)
        # [batch_size, 512, 3, 1]
        dense = conv_out.contiguous().view(conv_out.size()[0], -1)
        # [batch_soize, 1536]
        fc_out = self.fcblock1(dense)
        fc_out = self.fcblock2(fc_out)

        logit = self.activation(fc_out)

        return logit

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
















