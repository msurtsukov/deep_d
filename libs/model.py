import tensorflow as tf
from functools import partial
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
from libs.my_hm_lstm import StaticHMLSTMCell
import numpy as np


class Model:
    def __init__(self, args, training=True, decoding=False):
        self.args = args
        self.model = args.model
        self.vocab_size = args.vocab_size
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.rnn_size = args.rnn_size
        self.num_layers = args.num_layers
        self.boundary_symbols = args.boundary_symbols
        self.grad_clip = args.grad_clip
        self.training = training
        self.decoding = decoding
        if not self.training and not self.decoding:
            self.seq_length = 1
            self.batch_size = 1

        self._shm = False

        if self.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif self.model == 'gru':
            cell_fn = rnn.GRUCell
        elif self.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif self.model == 'nas':
            cell_fn = rnn.NASCell
        elif self.model == 'shm':
            cell_fn = StaticHMLSTMCell
            self._boundary_symbols = self.boundary_symbols
            assert len(self.boundary_symbols) == self.num_layers - 1
            self._shm = True
        else:
            raise Exception("model type not supported: {}".format(self.model))

        if self._shm:
            self.cell = cell = StaticHMLSTMCell(self.rnn_size, [self.rnn_size] * self.num_layers)
        else:
            cells = []
            for _ in range(self.num_layers):
                cell = cell_fn(self.rnn_size)
                cells.append(cell)

            self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.input_data = tf.placeholder(
            tf.int32, [self.batch_size, self.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [self.batch_size, self.seq_length])
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        if self._shm:
            self.boundary_data = []
            for _ in range(self.num_layers - 1):
                self.boundary_data.append(tf.placeholder(
                    tf.float32, [self.batch_size, self.seq_length]))

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [self.rnn_size, self.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size])

        embedding = tf.get_variable("embedding", [self.vocab_size, self.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if self._shm:
            inputs = [inputs] + [tf.expand_dims(b_data, -1) for b_data in self.boundary_data]
            inputs = tf.concat(inputs, axis=2)

        inputs = tf.split(inputs, self.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop_rnn(prev, _, model):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            inp = tf.nn.embedding_lookup(embedding, prev_symbol)
            if model:
                model.loop_samples.append(inp)
            return inp

        if self._shm:
            def loop_shm(prev, _, model):
                prev = tf.matmul(prev, softmax_w) + softmax_b
                # prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                prev_symbol = tf.stop_gradient(tf.squeeze(tf.multinomial(prev, 1)))
                inp = [tf.nn.embedding_lookup(embedding, prev_symbol)]
                for b_s in self.boundary_symbols:
                    z = tf.expand_dims(tf.to_float(tf.equal(prev_symbol, b_s)), axis=-1)
                    inp.append(z)
                inp = tf.concat(inp, axis=1)
                model.loop_samples.append(prev_symbol)
                return inp

        if self._shm:
            loop = loop_shm
        else:
            loop = loop_rnn

        loop = partial(loop, model=self)
        self.loop_samples = []
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell,
                                                         loop_function=loop if (not training) and decoding else None,
                                                         scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.rnn_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        if self._shm:
            self.hidden_l2_norm = self.cell.hidden_states_l2_seq
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([self.batch_size * self.seq_length])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / self.batch_size / self.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def get_hidden_l2_norm(self, sess, vocab, phrase):
        hidden_norms = []
        if self._shm:
            state = sess.run(self.cell.zero_state(1, tf.float32))
            for char in phrase:
                x = np.zeros((1, 1))
                x[0, 0] = vocab[char]
                feed = {self.input_data: x}
                for sym_code, placeholder in zip(self.boundary_symbols, self.boundary_data):
                    feed[placeholder] = np.equal(x, sym_code).astype(np.float32)
                for i, h in enumerate(self.initial_state):
                    feed[h] = state[i]

                state, hidden_l2_norm = sess.run([self.final_state, self.hidden_l2_norm[-1]], feed)
                hidden_norms.append(hidden_l2_norm)
        return hidden_norms

    # def loop_sample(self, sess, chars, vocab, num, prime, pad=' '):
    #     state = sess.run(self.cell.zero_state(self.batch_size, tf.float32))
    #     if not isinstance(prime, list):
    #         prime = [prime] * self.batch_size
    #     else:
    #         assert len(prime) == self.batch_size

    def calculate_states(self, sess, transformer, phrases, pad=' '):
        assert len(phrases) == self.batch_size
        prefix = pad * self.seq_length
        phrases = [(prefix + p)[-self.seq_length:] for p in phrases]
        x = np.reshape(transformer.transform(''.join(phrases)), (self.batch_size, self.seq_length))

        state = sess.run(self.cell.zero_state(self.batch_size, tf.float32))
        feed = {self.input_data: x}
        if self._shm:
            for sym_code, placeholder in zip(self.boundary_symbols, self.boundary_data):
                feed[placeholder] = np.equal(x, sym_code).astype(np.float32)
            for i, h in enumerate(self.initial_state):
                feed[h] = state[i]
        else:
            feed[self.initial_state] = state
        [state] = sess.run([self.final_state], feed)
        return state

    def loop_sample(self, sess, transformer, state):
        assert self.decoding
        assert len(state) == self.num_layers * 3
        x = np.zeros((self.batch_size, self.seq_length), dtype=np.int32)

        feed = {self.input_data: x}
        if self._shm:
            bx = np.zeros((self.batch_size, self.seq_length), dtype=np.float32)
            for sym_code, placeholder in zip(self.boundary_symbols, self.boundary_data):
                feed[placeholder] = bx
            for i, h in enumerate(self.initial_state):
                feed[h] = state[i]
        else:
            feed[self.initial_state] = state
        out = sess.run(self.loop_samples, feed)
        out = np.stack(out, axis=1)
        out = np.vsplit(out, out.shape[0])
        out = [transformer.inverse_transform(np.squeeze(id_seq)) for id_seq in out]
        return out

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1, stop_sym=None):
        state = sess.run(self.cell.zero_state(self.batch_size, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((self.batch_size, 1))
            x[:, 0] = vocab[char]
            feed = {self.input_data: x}
            if self._shm:
                for sym_code, placeholder in zip(self.boundary_symbols, self.boundary_data):
                    feed[placeholder] = np.equal(x, sym_code).astype(np.float32)
                for i, h in enumerate(self.initial_state):
                    feed[h] = state[i]
            else:
                feed[self.initial_state] = state
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return int(np.searchsorted(t, np.random.rand(1) * s))

        ret = [prime] * self.batch_size
        char = [prime[-1]] * self.batch_size
        for n in range(num):
            x = np.zeros((self.batch_size, 1))
            x[:, 0] = [vocab[c] for c in char]
            feed = {self.input_data: x}
            if self._shm:
                for sym_code, placeholder in zip(self.boundary_symbols, self.boundary_data):
                    feed[placeholder] = np.equal(x, sym_code).astype(np.float32)
                for i, h in enumerate(self.initial_state):
                    feed[h] = state[i]
            else:
                feed[self.initial_state] = state
            probs, state = sess.run([self.probs, self.final_state], feed)
            char = []
            for p in probs:
                if sampling_type == 0:
                    sample = np.argmax(p)
                elif sampling_type == 2:
                    if char == ' ':
                        sample = weighted_pick(p)
                    else:
                        sample = np.argmax(p)
                else:  # sampling_type == 1 default:
                    sample = weighted_pick(p)

                pred = chars[sample]
                if pred == stop_sym:
                    break
                else:
                    try:
                        if pred in stop_sym:
                            break
                    except TypeError:
                        pass
                char.append(pred)
            new_ret = []
            for r, c in zip(ret, char):
                new_ret.append(r + c)
            ret = new_ret
        return ret
