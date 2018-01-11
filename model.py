import json
import numpy as np
import tensorflow as tf

class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    '''Convolutional LSTM (Long short-term memory unit) recurrent network cell.

    The class uses optional peep-hole connections, optional cell-clipping,
    optional normalization layer, and an optional recurrent dropout layer.

    Basic implmentation is based on tensorflow.

    Default LSTM Network implementation is based on:

        http://www.bioinf.jku.at/publications/older/2604.pdf

    Sepp Hochreiter, Jurgen Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

    Peephole connection implementation is based on:

        https://research.google.com/pubs/archive/43905.pdf

    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for large scale acoustic modeling.". 2014.

    Default Convolutional LSTM implementation is based on:

        https://arxiv.org/abs/1506.04214

    Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, Wang-chun Woo.
    'Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting'. 2015.

    Recurrent dropout is base on:
    
        https://arxiv.org/pdf/1603.05118

    Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth
    'Recurrent Dropout without Memory Loss'. 2016.

    Normalization layer is applied before interal nonlinearities.
    '''
    def __init__(self,
                 shape,
                 kernel,
                 depth,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 forget_bias=1.0,
                 activation=None,
                 normalize=None,
                 dropout=None,
                 reuse=None):
        '''Initialize the parameters for a ConvLSTM Cell.

        Args:
            shape: list of 2 integers, specifying the height and width 
                of the input tensor.
            kernel: list of 2 integers, specifying the height and width 
                of the convolutional window.
            depth: Integer, the dimensionality of the output space.
            use_peepholes: Boolean, set True to enable diagonal/peephole connections.
            cell_clip: Float, if provided the cell state is clipped by this value 
                prior to the cell output activation.
            initializer: The initializer to use for the weights.
            forget_bias: Biases of the forget gate are initialized by default to 1
                in order to reduce the scale of forgetting at the beginning of the training.
            activation: Activation function of the inner states. Default: `tanh`.
            normalize: Normalize function, if provided inner states is normalizeed 
                by this function.
            dropout: Float, if provided dropout is applied to inner states 
                with keep probability in this value.
            reuse: Boolean, whether to reuse variables in an existing scope.
        '''
        super(ConvLSTMCell, self).__init__(_reuse=reuse)

        tf_shape = tf.TensorShape(shape + [depth])
        self._output_size = tf_shape
        self._state_size = tf.nn.rnn_cell.LSTMStateTuple(tf_shape, tf_shape)

        self._kernel = kernel
        self._depth = depth
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._activation = activation or tf.nn.tanh
        self._normalize = normalize
        self._dropout = dropout

        self._w_conv = None
        if self._use_peepholes:
            self._w_f_diag = None
            self._w_i_diag = None
            self._w_o_diag = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        '''Run one step of ConvLSTM.

        Args:
            inputs: input Tensor, 4D, (batch, shape[0], shape[1], depth)
            state: tuple of state Tensor, both `4-D`, with tensor shape `c_state` and `m_state`.

        Returns:
            A tuple containing:

            - A '4-D, (batch, height, width, depth)', Tensor representing 
                the output of the ConvLSTM after reading `inputs` when previous 
                state was `state`.
                Here height, width is:
                    shape[0] and shape[1].
            - Tensor(s) representing the new state of ConvLSTM after reading `inputs` when
                the previous state was `state`. Same type and shape(s) as `state`.
        '''
        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(4)[3]
        if input_size.value is None:
            raise ValueError('Could not infer size from inputs.get_shape()[-1]')

        c_prev, m_prev = state
        inputs = tf.concat([inputs, m_prev], axis=-1)

        if not self._w_conv:
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope, initializer=self._initializer):
                kernel_shape = self._kernel + [inputs.shape[-1].value, 4 * self._depth]
                self._w_conv = tf.get_variable('w_conv', shape=kernel_shape, dtype=dtype)

        # i = input_gate, j = new_input, f = forget_gate, o = ouput_gate
        conv = tf.nn.conv2d(inputs, self._w_conv, (1, 1, 1, 1), 'SAME')
        i, j, f, o = tf.split(conv, 4, axis=-1)

        # Diagonal connections
        if self._use_peepholes and not self._w_f_diag:
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope, initializer=self._initializer):
                self._w_f_diag = tf.get_variable('w_f_diag', c_prev.shape[1:], dtype=dtype)
                self._w_i_diag = tf.get_variable('w_i_diag', c_prev.shape[1:], dtype=dtype)
                self._w_o_diag = tf.get_variable('w_o_diag', c_prev.shape[1:], dtype=dtype)

        if self._use_peepholes:
            f = f + self._w_f_diag * c_prev
            i = i + self._w_i_diag * c_prev
        if self._normalize is not None:
            f = self._normalize(f)
            i = self._normalize(i)
            j = self._normalize(j)

        j = self._activation(j)

        if self._dropout is not None:
            j = tf.nn.dropout(j, self._dropout)

        c = tf.nn.sigmoid(f + self._forget_bias) * c_prev + tf.nn.sigmoid(i) * j

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            o = o + self._w_o_diag * c
        if self._normalize is not None:
            o = self._normalize(o)
            c = self._normalize(c)

        m = tf.nn.sigmoid(o) * self._activation(c)

        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, m)
        return m, new_state


class Predicator(object):
    ''' Predict daily average amount of fine dust.

    Attributes:
        matrix_shape: list of 2 integers, specifying the height and width of input matrix.
        num_time: Integer, the number of time series as an input.
        out_time: Integer, the number of time series as an output.
        kernels: list of n-tuples, each elements specifying the height and width of n-th convolutional window.
        depths: list of n-integers, specifying the depth of the ConvLSTM Cells.
        learning_rate: Float, learning rate in Adam for optimizing loss function.
        beta1: Float, beta1 value in Adam for optimizing loss function.
        plc_x: tf.placeholder, x vector.
        plc_y: tf.placeholder, y vector.
        model: tf.Tensor, model.
        loss: tf.Tensor, loss function.
        metric: tf.Tensor, metric for evaluating model.
        optimize: tf.Tensor, optimize object.
        summary: tf.Tensor, tensor summary of the metric.
    '''
    def __init__(self, matrix_shape, num_time, out_time, kernels, depths, learning_rate, beta1):
        ''' Initializer
        Args:
            matrix_shape: list of 2 integers, specifying the height and width of input matrix.
            num_time: Integer, the number of time series as an input.
            out_time: Integer, the number of time series as an output.
            kernels: list of n-tuples, each elements specifying the height and width of n-th convolutional window.
            depths: list of n-integers, specifying the depth of the ConvLSTM Cells.
            learning_rate: Float, learning rate in Adam for optimizing loss function.
            beta1: Float, beta1 value in Adam for optimizing loss function.
        '''
        self._matrix_shape = matrix_shape
        self._num_time = num_time
        self._out_time = out_time
        self._kernels = kernels
        self._depths = depths

        self._learning_rate = learning_rate
        self._beta1 = beta1

        self.plc_x = tf.placeholder(tf.float32, [None, self._num_time] + self._matrix_shape)
        if self._out_time == 1:
            self.plc_y = tf.placeholder(tf.float32, [None] + self._matrix_shape)
        else:
            self.plc_y = tf.placeholder(tf.float32, [None, self._out_time] + self._matrix_shape)

        self.model = self._get_model()
        self.loss = self._get_loss()
        self.metric = self._get_metric()

        self.optimize = tf.train.AdamOptimizer(self._learning_rate, self._beta1).minimize(self.loss)
        self.summary = tf.summary.scalar('Metric', self.metric)
        self._ckpt = tf.train.Saver()

    def train(self, sess, x, y):
        sess.run(self.optimize, feed_dict={self.plc_x: x, self.plc_y: y})

    def inference(self, sess, obj, x, y):
        return sess.run(obj, feed_dict={self.plc_x: x, self.plc_y: y})

    def dump(self, sess, path):
        self._ckpt.save(sess, path + '.ckpt')

        with open(path + '.json', 'w') as f:
            dump = json.dumps(
                {
                    'matrix_shape': self._matrix_shape,
                    'num_time': self._num_time,
                    'out_time': self._out_time,
                    'kernels': self._kernels,
                    'depths': self._depths,
                    'learning_rate': self._learning_rate,
                    'beta1': self._beta1
                }
            )

            f.write(dump)

    @classmethod
    def load(cls, sess, path):
        with open(path + '.json') as f:
            param = json.loads(f.read())

        model = cls(
            param['matrix_shape'],
            param['num_time'],
            param['out_time'],
            param['kernels'],
            param['depths'],
            param['learning_rate'],
            param['beta1']
        )
        model._ckpt.restore(sess, path + '.ckpt')

        return model

    def _get_model(self):
        assert len(self._kernels) == len(self._depths), 'Number of kernels and depths must be same.'
        
        shape = [-1, self._matrix_shape[0], self._matrix_shape[1], 1]
        inputs = tf.transpose(tf.reshape(self.plc_x, shape), [1, 0, 2, 3, 4])
        
        states = []
        hiddens = [inputs]

        for kernel, depth in zip(self._kernels, self._depths):
            cell = ConvLSTMCell(self._matrix_shape, kernel, depth)
            h, s = tf.nn.dynamic_rnn(cell, hiddens[-1], self._num_time, time_major=True)

            states.append(s)
            hiddens.append(h)

        if self._out_time == 1:
            concat = tf.concat([h[-1] for h in hiddens[1:]], axis=-1)
            flatten = tf.layers.conv2d(concat, 1, (1, 1), (1, 1))

            result = tf.reshape(flatten, [-1] + self._matrix_shape)
        else:
            hiddens = [tf.zeros((tf.shape(self.plc_x)[0], ))]
            for kernel, depth, state in zip(self._kernels, self._depths, states):
                cell = ConvLSTMCell(self._matrix_shape, kernel, depth)
                h, s = tf.nn.dynamic_rnn(cell, hidens[-1], self._out_time, initial_state=state, time_major=True)


        return result

    def _get_loss(self):
        return tf.reduce_mean(tf.square(self.plc_y - self.model))

    def _get_metric(self):
        return self.loss

    def _decode_loop(self):
        pass


class Batch(object):
    def __init__(self, x, y, batch_size):
        self.total_x = x
        self.total_y = y
        self.batch_size = batch_size

        self.iter_per_epoch = len(x) // batch_size
        self.epochs_completed = 0

        self._iter = 0

    def __call__(self):
        start = self._iter * self.batch_size
        end = (self._iter + 1) * self.batch_size

        batch_x = self.total_x[start:end]
        batch_y = self.total_y[start:end]

        self._iter += 1
        if self._iter == self.iter_per_epoch:
            self.epochs_completed += 1
            self._iter = 0

        return batch_x, batch_y
