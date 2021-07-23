import torch
import torch.nn as nn


class Net(nn.Module):
    """
    This is just so I can see
    how much improvement from stacking CNN's ontop of or before LSTM.
    """
    def __init__(self,params):
        super(Net, self).__init__()

        self.lstm = nn.LSTM(params.input_dim, params.hidden_dim, params.lstm_n_layers,
                             batch_first=True)

        self.linear = nn.Linear(params.linear_in, params.linear_out)
        self.activation = nn.LogSoftmax(dim=1)

        self.apply(self._init_weights)


    def forward(self, input):
        """
        Input: (batch_size, time_length, mel_filters)
        """
        lstm_out, _ = self.lstm(input)

        # (batch_size, time_length, lstm hidden channels)

        sentiment = lstm_out[:,-1]   # got better performance just looking at sentiment, rather than whole
        # [batch_size, 128]
        logit = self.linear(sentiment)

        return logit

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
















