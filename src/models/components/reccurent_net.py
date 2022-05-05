import math

import torch
import torch.nn as nn


class NaiveCustomLSTM(nn.Module):
    """A naive implementation of a LSTM network.

    f_t = sigmoid(U_f * x_t + V_f * h_{t-1} + b_f)
    i_t = sigmoid(U_i * x_t + V_i * h_{t-1} + b_i)
    o_t = sigmoid(U_o * x_t + V_o * h_{t-1} + b_o)

    g_t = tanh(U_g * x_t + V_g * h_{t-1} + b_g)
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # --- input gate
        self.U_i = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.V_i = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(hidden_dim))

        # --- forget gate
        self.U_f = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.V_f = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim))

        # --- output gate
        self.U_o = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.V_o = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(hidden_dim))

        # --- candidate gate (c_t)
        self.U_g = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.V_g = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.Tensor(hidden_dim))

        self.__init_weights()

    def __init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_tensor, cur_state):
        bs_size, seq_size = input_tensor.size()
        hidden_seq = []

        if cur_state is None:
            h_t, c_t = (
                torch.zeros(bs_size, self.hidden_dim),
                torch.zeros(bs_size, self.hidden_dim),
            )
        else:
            h_t, c_t = cur_state

        for t in range(seq_size):
            x_t = input_tensor[:, t, :]
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            g_t = torch.tanh(x_t @ self.U_g + h_t @ self.V_g + self.b_g)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        # reshape
        # TODO: Incomplete code
        pass


class SimpleRNN(nn.Module):
    def __init__(self, in_size: int = 59, hidden_size: int = 256, out_size: int = 18):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.in2hidden = nn.Linear(in_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(in_size + hidden_size, out_size)

    def forward(self, input_tensor, hidden_state=None):
        if hidden_state is None:
            hidden_state = torch.zeros(1, self.hidden_size)  # initial hidden state is zero

        combined = torch.cat((input_tensor, hidden_state), 1)
        hidden = torch.sigmoid(self.in2hidden(combined))
        output = self.in2output(combined)
        return output, hidden

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))
