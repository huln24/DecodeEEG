import torch.nn as nn
import torch

class MyBase(nn.Module):
  """MyBase.

    Arguments
    ---------
    input_size: int
        Number of input channels.
    out_size: int
        Number of output channels. 
    dropout: float
        Dropout probability.
    kernel_height: int
        Kernel size height value.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 1, 32, 200])
    #>>> model = MyBase(input_size=1, out_size=64, dropout=0.5, kernel_size=64)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1, 1, 64, 99])
    """
  def __init__(self, 
    input_size, 
    out_size, 
    dropout, 
    kernel_height):
    super().__init__()
    self.cnn = nn.Sequential(
      nn.Conv2d(in_channels=input_size, out_channels=out_size, kernel_size=(kernel_height, 7), stride=(1, 1), padding=(0, 2)),
      nn.BatchNorm2d(out_size),
      nn.Dropout(dropout),
      nn.MaxPool2d((1, 2))
    )

  def forward(self, x):
    """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
    """
    x = self.cnn(x)
    x = x.squeeze(dim=2).unsqueeze(dim=1)
    return x

class DeepCNN(nn.Module):
  """DeepCNN.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    depth: int
        Number of additional convolutional layers to perform.
    deep_cnn_out_size: int
        Number of output channels for deep convolutional layers.
    dropout: float
        Dropout probability.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = DeepCNN(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """
  def __init__(
    self,
    input_shape=(32, 500, 22, 1),
    depth=5,
    deep_cnn_out_size=32,
    dropout=0.25
    ):
    super().__init__()
    T = input_shape[1]
    C = input_shape[2]
    
    # DEEP CONVOLUTIONAL MODULE
    self.deep_cnn = torch.nn.Sequential(
      MyBase(1, deep_cnn_out_size, dropout, kernel_height=C),
      *[MyBase(1, deep_cnn_out_size, dropout, kernel_height=deep_cnn_out_size) for i in range(depth)]
    )

    # find input size for dense module
    out = self.deep_cnn(
      torch.ones((1, T, C, 1)).permute(0, 3, 2, 1)
    )
    dense_input_size = self._num_flat_features(out)

    # DENSE MODULE
    self.dense_module = torch.nn.Sequential(
      nn.Flatten(),
      nn.Linear(dense_input_size, 4),
      nn.LogSoftmax(dim=1)
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
    x = x.permute(0, 3, 2, 1)                  # (bs, 1, 22, 500)
    x = self.deep_cnn(x)                       # (bs, 1, 32, 1)
    return self.dense_module(x)
