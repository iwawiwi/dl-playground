from typing import Tuple, Union

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, kernel_size: Union[int, Tuple[int, int]], bias: bool
    ) -> None:
        """Initialize ConvLSTM cell

        Parameters
        ----------
        input_dim : int
            Number of channels of input tensor.
        hidden_dim : int
            Number of channels of hidden state.
        kernel_size : (int, int)
            Size of the convolutional kernel.
        bias : bool
            Whether or not to add the bias.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = (
            kernel_size // 2
            if isinstance(kernel_size, int)
            else (kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bias = bias

        self.conv = nn.Conv2d(
            self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width),
            torch.zeros(batch_size, self.hidden_dim, height, width),
        )


class ConvLSTM(nn.Module):
    """ConvLSTM module
    Parameters:
        input_dim: Number of channels of input tensor.
        hidden_dim: Number of channels of hidden state.
        kernel_size: Size of the convolutional kernel.
        num_layers: Number of stacked ConvLSTM layers.
        batch_first: Whether or not the first dimension represents the batch.
        bias: Whether or not to add the bias.
        return_all_layers: Whether or not to return outputs of all layers.

    Inputs:
        A tensor of size (B, T, C, H, W) for (batch, time, channel, height, width) or (T, B, C, H, W) for (time, batch, channel, height, width).

    Outputs:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of list of length T of each output
            1 - last_state_list is the list of last state, each element is a tuple of two elements (h, c) for hidden state and memory/cell state

    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_state = convlstm(x)
        >> h = last_state[0][0]  # get hidden state from last layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Union[int, Tuple[int, int]],
        num_layers: int,
        batch_first: bool = False,
        bias: bool = True,
        return_all_layers: bool = False,
    ) -> None:
        super().__init__()
        self.__check_kernel_size_consistency(kernel_size)

        # make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self.__extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self.__extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError(f"Inconsistent list length {len(kernel_size)} vs. {num_layers}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(cur_input_dim, self.hidden_dim[i], self.kernel_size[i], self.bias)
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters:
        -----------
        input_tensor:
            5D tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
            None. TODO: implement stateful

        Returns:
        --------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self.__init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def __init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    def __check_kernel_size_consistency(self, kernel_size):
        if not (
            isinstance(kernel_size, Tuple)
            or isinstance(kernel_size, list)
            and all([isinstance(elem, Tuple) for elem in kernel_size])
        ):
            raise ValueError(
                "`kernel_size` must be tuple or list with length equal to `num_layers`."
            )

    def __extend_for_multilayer(self, param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(
            input_dim=in_chan, hidden_dim=nf, kernel_size=(3, 3), bias=True
        )

        self.encoder_2_convlstm = ConvLSTMCell(
            input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True
        )

        self.decoder_1_convlstm = ConvLSTMCell(
            input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True  # nf + 1
        )

        self.decoder_2_convlstm = ConvLSTMCell(
            input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True
        )

        self.decoder_CNN = nn.Conv3d(
            in_channels=nf, out_channels=1, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(
                input_tensor=x[:, t, :, :], cur_state=[h_t, c_t]
            )  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(
                input_tensor=h_t, cur_state=[h_t2, c_t2]
            )  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(
                input_tensor=encoder_vector, cur_state=[h_t3, c_t3]
            )  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(
                input_tensor=h_t3, cur_state=[h_t4, c_t4]
            )  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(
            x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4
        )

        return outputs
