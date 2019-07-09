import tensorflow as tf
import numpy as np

class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.
  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=True, peephole=True, data_format='channels_last', reuse=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    c, h = state

    x = tf.concat([x, h], axis=self._feature_axis)
    n = x.shape[-1].value
    m = 4 * self._filters if self._filters > 1 else 4
    W = tf.get_variable('kernel', self._kernel + [n, m])
    y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)
    if not self._normalize:
      y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    if self._peephole:
      i += tf.get_variable('W_ci', c.shape[1:]) * c
      f += tf.get_variable('W_cf', c.shape[1:]) * c

    if self._normalize:
      j = tf.contrib.layers.layer_norm(j)
      i = tf.contrib.layers.layer_norm(i)
      f = tf.contrib.layers.layer_norm(f)

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)

    if self._peephole:
      o += tf.get_variable('W_co', c.shape[1:]) * c

    if self._normalize:
      o = tf.contrib.layers.layer_norm(o)
      c = tf.contrib.layers.layer_norm(c)

    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state


class ConvGRUCell(tf.nn.rnn_cell.RNNCell):
  """A GRU cell with convolutions instead of multiplications."""

  def __init__(self, shape, filters, kernel, activation=tf.tanh, normalize=True, data_format='channels_last', reuse=None):
    super(ConvGRUCell, self).__init__(_reuse=reuse)
    self._filters = filters
    self._kernel = kernel
    self._activation = activation
    self._normalize = normalize
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def call(self, x, h):
    channels = x.shape[self._feature_axis].value

    with tf.variable_scope('gates'):
      inputs = tf.concat([x, h], axis=self._feature_axis)
      n = channels + self._filters
      m = 2 * self._filters if self._filters > 1 else 2
      W = tf.get_variable('kernel', self._kernel + [n, m])
      y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
      if self._normalize:
        r, u = tf.split(y, 2, axis=self._feature_axis)
        r = tf.contrib.layers.layer_norm(r)
        u = tf.contrib.layers.layer_norm(u)
      else:
        y += tf.get_variable('bias', [m], initializer=tf.ones_initializer())
        r, u = tf.split(y, 2, axis=self._feature_axis)
      r, u = tf.sigmoid(r), tf.sigmoid(u)

    with tf.variable_scope('candidate'):
      inputs = tf.concat([x, r * h], axis=self._feature_axis)
      n = channels + self._filters
      m = self._filters
      W = tf.get_variable('kernel', self._kernel + [n, m])
      y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
      if self._normalize:
        y = tf.contrib.layers.layer_norm(y)
      else:
        y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
      h = u * h + (1 - u) * self._activation(y)

    return h, h

class AE_Down(object):
    def __init__(self, xs, n_nodes, n_channels, shrink):
        self.name = name # naming the modality
        self.channels = n_channels
        self.shrink = shrink
        self.num_layers = 3 #pending
        self.conv_len = conv_eln #pending
    def ED_TCN(self, xs):
        """
        input: spatial or temporal cnn feature vectors (1D)
        return: compressed feat (the input of binary layer)
        """
        print("AutoEncoder layers")
        
        # encoder TC layers: input - padding - TCL - activation - max pooling - output
        for layer in range(num_layers):
            if causal: # advoid using further time steps
                xs = tf.image.resize_image_with_crop_or_pad(xs, 1, self.conv_len//2.0) # h w
            with tf.name_scope('Conv1d') as scope:
                kernel = tf.get_variable('{}/weight'.format(scope), shape=[3,3,64],
                                     initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01))
                biases = tf.get_variable('{}/bias'.format(scope), shape=[64],
                                   initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv1d(img, kernel, [1,1,1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                conv1 = tf.nn.relu(out, name='{}'.format(scope))
            
            if causal: # crop the 1st dim
                conv1 = tf.image.resize_image_with_crop_or_pad(conv1, 1, self.conv_len//2.0)

            dropout = tf.nn.dropout(conv1, rate=0.3)
            
            tanh = self.adaptive_tanh(dropout, lambd, alpha)
                 
            pool = tf.nn.max_pool1d(tanh,
                                    ksize=[1,3,1],
                                    strides=[1,2,1],
                                    paddings='SAME',
                                    name='pool_'+str(layer))
            xs = pool
        feat = pool
        return feat

    def DE_TCN(self, xs):
        """
        input: binary
        return: reconstructed
        """
        print('Decoder Layers')
            
        # decoder TC layers: input - upsampling - padding - TCL - activation - output
        for layer in range(num_layers):
            xs = tf.image.resize_images(xs, len(xs)*2, method=BILINEAR)
            if causel:
                xs = tf.image.resize_image_with_crop_or_pad(xs, 1, self.conv_len//2.0)
            with tf.name_scope('Conv1d') as scope:
                kernel = tf.get_variable('{}/weight'.format(scope), shape=[3,3,64],
                                         initializer=tf.random.normal_initializer(mean=0.0,stddev=0.01)
                biases = tf.get_variable('{}/bias'.format(scope), shape=[64],
                                         initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv1d(xs, kernel, [1,1,1], padding='SAME')
                out = tf.nn.bias_add(conv, biases)
                conv1 = tf.nn.relu(out, name='{}'.format(scope))
            if causel:
                conv1 = tf.image.resize_iamge_with_crop_or_pad(conv1, 1, self.conv_len//2.0)
            dropout = tf.nn.dropout(conv1, rate=0.3)
            
            tanh = self.adaptive_tanh(dropout, lambd, alpha)
            
        return rec_xs
    
    def adaptive_tanh(self, s, lambd, alpha):
        # nonlinear adaptive tanh: f(s)=tanh(log(s+1))+l2-norm
        
        f1 = tf.nn.tanh(s)
        f2 = tf.norm(alpha, ord=2, axis=0)
        f = f1 + lambd*f2
        return f
