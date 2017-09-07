import codecs
import os
import re
from six.moves import cPickle
import json
import numpy as np
import tensorflow as tf
from libs.model import Model
import libs.meaning_discrimination as me
from collections import namedtuple


symbs = re.compile(r"[^А-Яа-я:!\?,\.\"— - \n]")


def text_preprocess(text):
    text = re.sub("—", "-", text)
    text = re.sub(symbs, "", text)
    text = text.lower()
    return text


def compute_boundary(x, vocab, stop_syms):
    zs = []
    for stop_sym in stop_syms:
        z = np.equal(x, vocab[stop_sym]).astype(np.float32)
        zs.append(z)
    return zs


class TextLoader:
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8', shm=True, stop_syms=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
        self.shm = shm
        self.stop_syms = stop_syms or (" ", "\n")
        self.transformer = Token2IDTransformer(as_string=True)

        self.tensor = None
        self.num_batches = None
        self.x_batches = None
        self.y_batches = None
        self.z_batches = None
        self.pointer = None

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.json")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def load_vocabulary(self, vocab_file):
        with open(vocab_file, 'r', encoding=self.encoding) as f:
            vocab = json.load(f)
        self.transformer.set_vocab(vocab)

    def preprocess(self, input_file, vocab_file, tensor_file, preprocess_f=None):
        preprocess_f = preprocess_f or text_preprocess
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        if preprocess_f:
            data = text_preprocess(data)

        self.load_vocabulary(vocab_file)
        self.tensor = self.transformer.transform(data)
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        self.load_vocabulary(vocab_file)
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.z_batches = []
        zs = compute_boundary(xdata, vocab=self.transformer.vocab, stop_syms=self.stop_syms)
        self.z_batches = [np.split(z.reshape(self.batch_size, -1), self.num_batches, 1) for z in zs]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y, zs = self.x_batches[self.pointer], self.y_batches[self.pointer], [z[self.pointer] for z in self.z_batches]
        self.pointer += 1
        if self.shm:
            return x, y, zs
        else:
            return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


def load_transformer(load_dir):
    with open(os.path.join(load_dir, 'transformer.pkl'), 'rb') as f:
        transformer = cPickle.load(f)
    return transformer


def load_dictionary(load_dir):
    with open(os.path.join(load_dir, 'words_dictionary.txt'), 'r', encoding="utf-8") as f:
        dictionary = f.read().split('_')
    dictionary = dict(zip(dictionary, [1] * len(dictionary)))
    return dictionary


def load_model(save_dir, sess, training=False, decoding=False, **kwargs):
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)

    saved_args.__dict__.update(kwargs)

    model = Model(saved_args, training=training, decoding=decoding)
    with sess.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
    return model


def load_dis(load_dir, dis_type, file_name=None):
    assert dis_type in ("believability", "style", "meaning")

    if dis_type == "meaning":
        tokenizer, morph, w2v = me.load(load_dir)
        Meaning = namedtuple("Meaning", "tokenizer morph w2v")
        return Meaning(tokenizer, morph, w2v)

    from keras.models import load_model as load_m
    if not file_name:
        file_name = 'discriminator_' + dis_type + '_cnn_model.h5'
    path = os.path.join(load_dir, file_name)
    try:
        f = open(path)
        f.close()
    except Exception:
        raise Exception('No file ' + path)
    return load_m(path)


def top_best(list_of_pp_tuples, topn, remove_duplicates=True):
    if isinstance(topn, float):
        topn = max(min(1., topn), 0.)
        topn = int(len(list_of_pp_tuples) * topn)

    if remove_duplicates:
        list_of_pp_tuples = list(set(list_of_pp_tuples))

    sorted_out = sorted(list_of_pp_tuples, key=lambda x: x[1], reverse=True)[:topn]
    return sorted_out


class Token2IDTransformer:
    """
    Class encapsulating token and id mutual mapping
    """
    def __init__(self, as_string=True, default_token=None):
        self.as_string = as_string

        self.tokens = None
        self.vocab = None
        self.vocab_size = None
        self.token_idx = None
        self.idx_token = None
        self.default_id = None
        self.default_token = default_token

    def fit(self, tokens_array):
        """
        tokens_array should be an array of tokens or a string
        in case string -> individual chars are assumed to be tokens
        """
        if isinstance(tokens_array, str):
            tokens_array = list(tokens_array)
            self.as_string = True
        assert isinstance(tokens_array, list)

        self.set_vocab(sorted(list(set(tokens_array))))
        return self

    def set_vocab(self, tokens):
        assert isinstance(tokens, list)
        self.tokens = tokens
        self.vocab = dict((t, i) for i, t in enumerate(sorted(list(set(tokens)))))
        if self.default_token:
            if self.default_token in self.vocab.keys():
                self.default_id = self.vocab[self.default_token]
            else:
                self.default_id = len(self.vocab)
                self.tokens.append(self.default_token)
                self.vocab[self.default_token] = self.default_id
        else:
            self.default_id = None
        self.default_id = self.vocab[self.default_token] if self.default_token else None
        self.vocab_size = len(self.vocab)
        self.token_idx = self.vocab
        self.idx_token = dict((t, i) for i, t in self.vocab.items())

    def t2i_map_f(self, t):
        return self.token_idx.get(t, self.default_id)

    def i2t_map_f(self, i):
        return self.idx_token.get(i, self.default_token)

    def transform(self, text, one_hot=False):
        try:
            rez = np.array(list(map(self.t2i_map_f, text)), dtype=np.int)
        except TypeError:
            rez = self.t2i_map_f(text)
        if one_hot:
            from keras.utils.np_utils import to_categorical
            rez = to_categorical(rez)
        return rez

    def inverse_transform(self, idxs, one_hot=False):
        if one_hot:
            idxs = np.argmax(idxs, axis=-1)
        try:
            rez = list(map(self.i2t_map_f, idxs))
        except TypeError:
            rez = self.i2t_map_f(idxs)
        return rez

    def fit_transform(self, tokens_array, *args, **kwargs):
        self.fit(tokens_array)
        return self.transform(tokens_array, *args, **kwargs)


def pad(array, to_len, with_what):
    res = 0
    if len(array) > to_len:
        res = array[:to_len]
    elif len(array) < to_len:
        res = [with_what] * (to_len - len(array)) + list(array)
    else:
        res = array
    return np.array(res)


import re
pat = re.compile(r'([a-я:, "]+[?!.]+)')
w_pat = re.compile(r'[а-я]+')


def check_sent(seq):
    return len(seq) < 200


def filter_sequence(seq, dictionary):
    candidates = pat.findall(seq)
    selected = []
    for cand in candidates:
        cand = cand.strip()
        words = w_pat.findall(cand)
        keep = True
        for word in words:
            if not dictionary.get(word, 0):
                keep = False
        if keep:
            selected.append(cand)
    return selected


def split_data_into_correct_batches(text1_indexes, text2_indexes, n_batches,
                                    max_len, make_equal_folding=True):
    prime_number = 2147483647

    X = np.zeros((n_batches, max_len), dtype=np.int64)
    Y = np.zeros((n_batches,), dtype=np.int64)

    choose_from_first = True
    index1 = 0
    index2 = 0
    for i in range(n_batches):
        if make_equal_folding:
            if choose_from_first:
                index1 = (index1 + prime_number) % (len(text1_indexes))
                X[i, :] = text1_indexes[index1]
                Y[i] = 0
            else:
                index2 = (index2 + prime_number) % (len(text2_indexes))
                X[i, :] = text2_indexes[index2]
                Y[i] = 1

            choose_from_first = not choose_from_first
        else:
            index1 = (index1 + prime_number) % (len(text1_indexes) + len(text2_indexes))
            if index1 < len(text1_indexes) - max_len + 1:
                X[i, :] = text1_indexes[index1]
                Y[i] = 0
            else:
                index2 = index1 - (len(text1_indexes))
                X[i, :] = text2_indexes[index2]
                Y[i] = 1
    return X, Y


def split_data_into_correct_batches_for_stateful_rnn(array, batch_size, max_len):
    """Array is a numpy array of indices, first dim is sequence dim, others may or may not present;

For input:
- array = np.array([0, 1, 2, 3, ..., 99]),
- batch_size = 4,
- max_len = 8

output X is:
array([[ 0,  1,  2,  3,  4,  5,  6,  7],
       [25, 26, 27, 28, 29, 30, 31, 32],
       [50, 51, 52, 53, 54, 55, 56, 57],
       [75, 76, 77, 78, 79, 80, 81, 82],

       [ 8,  9, 10, 11, 12, 13, 14, 15],
       [33, 34, 35, 36, 37, 38, 39, 40],
       [58, 59, 60, 61, 62, 63, 64, 65],
       [83, 84, 85, 86, 87, 88, 89, 90],

       [16, 17, 18, 19, 20, 21, 22, 23],
       [41, 42, 43, 44, 45, 46, 47, 48],
       [66, 67, 68, 69, 70, 71, 72, 73],
       [91, 92, 93, 94, 95, 96, 97, 98]])

Such packing where beginning of new batch is the continuation of previous is needed for correct
training of stateful rnn"""

    step = array.shape[0] // batch_size
    seqs_num = (step - 1) // max_len  # make sure there is always an y ahead
    tail_dims = list(array.shape[1:])

    r_text = array.reshape([batch_size, len(array) // batch_size] + tail_dims)
    X = r_text[:, :seqs_num * max_len].reshape([batch_size, seqs_num, max_len] + tail_dims)
    X = X.reshape([batch_size * seqs_num, max_len] + tail_dims, order="F")

    y = r_text[:, 1:seqs_num * max_len + 1].reshape([batch_size, seqs_num, max_len] + tail_dims)
    y = y.reshape([batch_size * seqs_num, max_len] + tail_dims, order="F")
    return X, y


def beam_search(predict_f, seed, top_k, seq_len=1, temp_seq=1.0, return_full_tree=False):
    """
predict_f([seed]) - функция сэмплинга следующего токена (на входе массив последовательностей,
                                                                      на выходе массив вероятностей)
seed - последовательность предыдущих токенов
top_k - сколько токенов с наибольшей вероятностью учитывать, при сэмплинге следующего
seq_len - длина последовательности, которую сэмплить
temp_seq - температура софтмакса конечных последовательностей
return_full_tree - вернуть деревья последовательностей и их вероятностей целиком
    """
    assert top_k ** seq_len < 10000, 'complexity is O(top_k ** seq_len)'

    s = list(seed.shape)  # without batch_size
    s[0] = 0
    sequences = [[] for i in range(seq_len + 1)]
    probabilities = [[] for i in range(seq_len + 1)]
    sequences[0].append(np.zeros(s, dtype=np.int))
    probabilities[0].append(1.)

    seed_len = len(seed)
    for i in range(seq_len):
        seed_samples = []
        for seq in sequences[i]:
            seed_samples.append(np.concatenate((seed[i:], seq)))

        preds_array = predict_f(seed_samples)
        for seq, prob, preds in zip(sequences[i], probabilities[i], preds_array):
            arg_preds = np.argsort(preds)[-top_k:]
            probs = preds[arg_preds]

            added_seqs = [np.concatenate((seq, p)) for p in np.split(arg_preds, top_k)]
            sequences[i + 1] += added_seqs
            probabilities[i + 1] += [a[0] for a in np.split(probs * prob, top_k)]

    if return_full_tree:
        return sequences, probabilities
    else:
        log_probs = np.log(np.array(probabilities[-1]) + 1e-10) / temp_seq
        probs = np.exp(log_probs)
        final_preds = np.random.multinomial(1, probs / np.sum(probs), 1)
        return sequences[-1][np.argmax(final_preds)]


def predict_f_for_stateful_rnn(rnn, batch_shape, seed, temperature=1.0):
    x = np.zeros((batch_shape,))
    assert isinstance(seed, (list, tuple))
    preds_array = []
    for batch_begin in range(0, len(seed), batch_shape[0]):
        for i, s in enumerate(seed[batch_begin:batch_begin + batch_shape[0]]):
            x[i] = s
        preds = rnn.predict(x, verbose=0)[:i + 1, -1, :]
        preds = np.log(preds + 1e-10) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds, axis=-1)[:, None]
        preds_array += list(map(np.squeeze, np.vsplit(preds, i + 1)))
    return preds_array
