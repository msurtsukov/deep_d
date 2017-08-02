import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import RNNCell, _linear
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops


def _norm(inp, scope):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(1.)
    beta_init = init_ops.constant_initializer(0.)
    with vs.variable_scope(scope):
        # Initialize beta and gamma for use by layer_norm.
        vs.get_variable("gamma", shape=shape, initializer=gamma_init)
        vs.get_variable("beta", shape=shape, initializer=beta_init)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized


class SubStaticHMLSTM:
    def __init__(self, num_units, activation, layer_norm=True):
        self._num_units = num_units
        self._activation = activation
        self._layer_norm = layer_norm

    def __call__(self, c_prev, h_prev, h_down, h_top_prev, z, z_prev, z_down, scope=None):
        with tf.name_scope(scope or type(self).__name__):
            fiog = _linear([h_prev * (1 - z_prev),
                            h_top_prev * z_prev,
                            h_down * z_down],
                           4 * self._num_units, True)
            if self._layer_norm:
                fiog = _norm(fiog, "layer_norm")

            fio, g = tf.split(fiog, [3 * self._num_units, self._num_units],
                              axis=1, name="split_to_function_arguments")

            fio = tf.sigmoid(fio)
            g = tf.tanh(g)
            f, i, o = tf.split(fio, 3, axis=1, name='split_gates')

            with tf.name_scope('boundary_operations'):
                update_flag = tf.to_float(tf.logical_and(tf.equal(z_prev, 0., name="z_prev_is_0"),
                                                         tf.equal(z_down, 1., name="z_down_is_1"),
                                                         name="logical_and_in_update_flag"),
                                          name="to_float_in_update_flag")
                copy_flag = tf.to_float(tf.logical_and(tf.equal(z_prev, 0., name="z_prev_is_0"),
                                                       tf.equal(z_down, 0., name="z_down_is_0"),
                                                       name="logical_and_in_copy_flag"),
                                        name="to_float_in_update_flag")
                flush_flag = tf.to_float(tf.equal(z_prev, 1., name="z_prev_is_1"),
                                         name="to_float_in_flush_flag")

                c = update_flag * self._update_op(f, c_prev, i, g) + \
                    flush_flag * self._flush_op(i, g) + \
                    copy_flag * self._copy_op(c_prev)

                h = copy_flag * h_prev + (1 - copy_flag) * o * self._activation(c)

        return h, c, z

    @staticmethod
    def _update_op(f, c_prev, i, g):
        return f * c_prev + i * g

    @staticmethod
    def _copy_op(c_prev):
        return c_prev

    @staticmethod
    def _flush_op(i, g):
        return i * g


class StaticHMLSTMCell(RNNCell):
    """Static HM_LSTM"""
    def __init__(self, num_units, units_per_abstraction, output_activation=None,
                 level_activation=None, reuse=None, layer_norm=True):
        super(StaticHMLSTMCell, self).__init__(_reuse=reuse)
        self._levels_of_abstraction = len(units_per_abstraction)
        self._units_per_abstraction = list(units_per_abstraction)
        self._num_units = num_units
        self._output_activation = output_activation
        self._level_activation = level_activation or tf.tanh
        self._layer_norm = layer_norm
        self.hidden_states_l2_seq = []

    def call(self, inputs, state):
        input_size = int(inputs.get_shape()[-1]) - self._levels_of_abstraction + 1
        inputs = tf.split(inputs, [input_size] + [1] * (self._levels_of_abstraction - 1), 1)

        h_0, z_0 = inputs[0], 1.  # z_0 is expected to be 1 always
        batch_size = h_0.get_shape()[0]

        prev_h_states = list(state[:self._levels_of_abstraction]) + [tf.constant(0, shape=(batch_size, 0))]
        prev_c_states = list(state[self._levels_of_abstraction:2*self._levels_of_abstraction])
        prev_z_states = list(state[2*self._levels_of_abstraction:])

        sub_h_states = []
        sub_c_states = []
        sub_z_states = inputs[1:] + [tf.constant(0., shape=(batch_size, 1), dtype=tf.float32)]  # z_0

        new_hidden_states_l2 = []
        with tf.variable_scope("level_0"):
            level_1 = SubStaticHMLSTM(self._units_per_abstraction[0], activation=self._level_activation,
                                      layer_norm=self._layer_norm)
            h_1, c_1, z_0 = level_1(prev_c_states[0], prev_h_states[0], h_0, prev_h_states[1],
                                    sub_z_states[0], prev_z_states[0], z_0)
            sub_h_states.append(h_1)
            sub_c_states.append(c_1)
            new_hidden_states_l2.append(self._h_state_l2(h_1))

        for i, num_units in enumerate(self._units_per_abstraction[1:]):
            i += 1
            with tf.variable_scope("level_{}".format(i)):
                level_i = SubStaticHMLSTM(self._units_per_abstraction[i], activation=self._level_activation,
                                          layer_norm=self._layer_norm)
                h_i, c_i, z_im1 = level_i(prev_c_states[i], prev_h_states[i], sub_h_states[i-1], prev_h_states[i+1],
                                          sub_z_states[i], prev_z_states[i], sub_z_states[i-1])

            sub_h_states.append(h_i)
            sub_c_states.append(c_i)
            new_hidden_states_l2.append(self._h_state_l2(h_i))
        self.hidden_states_l2_seq.append(tf.stack(new_hidden_states_l2, axis=1))

        with tf.variable_scope("output"):
            with tf.variable_scope("weighting"):
                concatenated_h = tf.concat(sub_h_states, axis=1)
                logits = _linear(concatenated_h, self._levels_of_abstraction, False)
                if self._layer_norm:
                    logits = _norm(logits, "layer_norm")
                g = tf.split(tf.sigmoid(logits), self._levels_of_abstraction, axis=1)
                weighted_hs = []
                for i in range(self._levels_of_abstraction):
                    with tf.variable_scope("level_{}".format(i)):
                        level_i_out_transformed = _linear(sub_h_states[i], self._num_units, True)
                        if self._layer_norm:
                            level_i_out_transformed = _norm(level_i_out_transformed, "layer_norm")
                        weighted_hs.append(g[i] * level_i_out_transformed)
            with tf.variable_scope("sum"):
                h_out = 0
                for h in weighted_hs:
                    h_out += h
            if self._output_activation:
                with tf.variable_scope("activation"):
                    h_out = self._output_activation(h_out)
        return h_out, sub_h_states + sub_c_states + sub_z_states

    @staticmethod
    def _h_state_l2(h):
        return tf.sqrt(tf.reduce_sum(tf.square(h), axis=1))

    @property
    def state_size(self):
        return tuple(self._units_per_abstraction * 2 + [1] * self._levels_of_abstraction)

    @property
    def output_size(self):
        return self._num_units

if __name__ == '__main__':
    pass
