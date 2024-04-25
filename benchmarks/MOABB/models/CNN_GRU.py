
import torch
import torch.nn as nn
import speechbrain as sb


class CNN_GRU(torch.nn.Module):
    """CNN_GRU.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    cnn_spatial_kernels:
        Number of output channels for spatial convolution.
    cnn_spatial_max_norm: float
        Kernel max norm of the 2d spatial depthwise convolution.
    cnn_spatial_pool: tuple
        Pool size and stride after the 2d spatial depthwise convolution.
    cnn_pool_type: string
        Pooling type.  
    cnn_dropout: float
        Dropout probability for spatial convolution.
    gru_hidden: int
        Number of hidden features in state h in GRU.
    gru_layers: int
        Number of recurrent layers.
    gru_dropout: float
        Dropout probability for GRU.
    dense_max_norm: float
        Weight max norm of the fully-connected layer.
    dense_n_neurons: int
        Number of output neurons.
    activation_type: str
        Activation function of the hidden layers.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = CNN_GRU(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """
    def __init__(
        self,
        input_shape=None,  # (1, T, C, 1)
        cnn_spatial_kernels=8,
        cnn_spatial_max_norm=1.0,
        cnn_spatial_pool=(4, 1),
        cnn_pool_type="max",
        cnn_dropout=0.5,
        gru_hidden=10,
        gru_layers=2,
        gru_dropout=0.25,
        dense_max_norm=0.25,
        dense_n_neurons=4,
        activation_type="elu",
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        if activation_type == "gelu":
            activation = torch.nn.GELU()
        elif activation_type == "elu":
            activation = torch.nn.ELU()
        elif activation_type == "relu":
            activation = torch.nn.ReLU()
        elif activation_type == "leaky_relu":
            activation = torch.nn.LeakyReLU()
        elif activation_type == "prelu":
            activation = torch.nn.PReLU()
        else:
            raise ValueError("Wrong hidden activation function")
        self.default_sf = 128  # sampling rate of the original publication (Hz)
        T = input_shape[1]
        C = input_shape[2]

        # CONVOLUTIONAL MODULE
        # Spatial depthwise convolution
        self.conv_module = torch.nn.Sequential(
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_spatial_kernels,
                kernel_size=(1, C),
                groups=1,
                padding="valid",
                bias=False,
                max_norm=cnn_spatial_max_norm,
                swap=True,
            ),
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels, momentum=0.01, affine=True,
            ),
            activation,
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_spatial_pool,
                stride=cnn_spatial_pool,
                pool_axis=[1, 2],
            ),
            torch.nn.Dropout(p=cnn_dropout)
        )

        # RNN MODULE
        self.h0 = nn.Parameter(torch.randn(1 * gru_layers, gru_hidden))
        self.gru = nn.GRU(
            input_size=cnn_spatial_kernels, 
            hidden_size=gru_hidden, 
            num_layers=gru_layers, 
            batch_first=True, 
            dropout=gru_dropout, 
            bidirectional=False
        )

        # Shape of intermediate feature maps
        out = self.conv_module(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        out, _ = self.gru(out.squeeze(2))
        dense_input_size = self._num_flat_features(out)
        
        # DENSE MODULE
        self.dense_module = torch.nn.Sequential(
            torch.nn.Flatten(),
            sb.nnet.linear.Linear(input_size=dense_input_size, n_neurons=dense_n_neurons, max_norm=dense_max_norm),
            torch.nn.LogSoftmax(dim=1)  
        )

    
    def _num_flat_features(self, x):
        """Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        x = self.conv_module(x)      # (bs, 125, 1, 16)
        x = x.squeeze(2)             # (bs, 125, 16) 
        h0 = torch.stack([self.h0 for _ in x], dim=1)
        x, _ = self.gru(x, h0)       # (bs, 125, gru_hidden)
        # x = x[:, -1, :]            # (bs, gru_hidden)
        x = self.dense_module(x)     # (bs, 4)
        return x

