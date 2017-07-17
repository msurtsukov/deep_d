from keras.utils.np_utils import to_categorical
import numpy as np


class Token2IDTransformer:
    '''
Class incapsulating token and id mutual mapping
    '''
    def fit(self, tokens_array):
        '''
tokens_array should be an array of tokens or a string
in case string -> individual chars are assumed to be tokens
        '''
        self.asstring = False
        if isinstance(tokens_array, str):
            tokens_array = list(tokens_array)
            self.asstring = True
        assert isinstance(tokens_array, list)

        self.vocab = sorted(list(set(tokens_array)))
        self.vocab_size = len(self.vocab)

        self.token_idx = dict((t, i) for i, t in enumerate(self.vocab))
        self.idx_token = dict((i, t) for i, t in enumerate(self.vocab))
        self.t2i_map_f = lambda t: self.token_idx[t]
        self.i2t_map_f = lambda i: self.idx_token[i]
        return self

    def transform(self, text, one_hot=False):
        rez = None
        try:
            rez = np.array(list(map(self.t2i_map_f, text)), dtype=np.int)
        except TypeError:
            rez = self.t2i_map_f(text)
        if one_hot:
            rez = to_categorical(rez)
        return rez

    def inverse_transform(self, idxs, one_hot=False):
        rez = None
        if one_hot:
            idxs = np.argmax(idxs, axis=-1)
        try:
            rez = list(map(self.i2t_map_f, idxs))
            if self.asstring:
                rez = ''.join(rez)
        except TypeError:
            rez = self.i2t_map_f(idxs)
        return rez

    def fit_transform(self, tokens_array, *args, **kwargs):
            self.fit(tokens_array, *args, **kwargs)
            return self.transform(tokens_array, *args, **kwargs)


def split_data_into_correct_batches_for_stateful_rnn(array, batch_size, max_len):
    '''Array is a numpy array of indices, first dim is sequence dim, others may or may not present;

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
training of stateful rnn'''

    step = array.shape[0] // batch_size
    seqs_num = (step - 1) // max_len # make sure there is always an y ahead
    tail_dims = list(array.shape[1:])
    
    r_text = array.reshape([batch_size, len(array) // batch_size] + tail_dims)
    X = r_text[:,:seqs_num * max_len].reshape([batch_size, seqs_num, max_len] + tail_dims)
    X = X.reshape([batch_size * seqs_num, max_len] + tail_dims, order="F")

    y = r_text[:,1:seqs_num * max_len + 1].reshape([batch_size, seqs_num, max_len] + tail_dims)
    y = y.reshape([batch_size * seqs_num, max_len] + tail_dims, order="F")
    return X, y


def deep_sample_seq(predict_f, seed, top_k, seq_len=1, temp_seq=1.0, return_full_tree=False):
    ''' Не помню как правильно называется, на семинарах рассказывали, когда сэмплишь не по одному 
токену, а сразу последовательность. Основываясь на условной зависимости следующего токена от предыдущего.
predict_f([seed]) - функция сэмплинга следующего токена (на входе массив последовательностей, 
                                                                      на выходе массив вероятностей)
seed - последовательность предыдущих токенов
top_k - сколько токенов с наибольшей вероятностью учитывать, при сэмплинге следующего
seq_len - длина последовательности, которую сэмплить
temp_seq - температура софтмакса конечных последовательностей
return_full_tree - вернуть деревья последовательностей и их вероятностей целиком
    '''
    assert top_k ** seq_len < 10000, 'complexity is O(top_k ** seq_len)'

    s = list(seed.shape) # without batch_size
    s[0] = 0
    sequences     = [[] for i in range(seq_len + 1)]
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
            sequences[i+1] += added_seqs
            probabilities[i+1] += [a[0] for a in np.split(probs * prob, top_k)]

    if return_full_tree:
        return sequences, probabilities
    else:
        log_probs = np.log(np.array(probabilities[-1]) + 1e-10) / temp_seq
        probs = np.exp(log_probs)
        final_preds = np.random.multinomial(1, probs / np.sum(probs), 1)
        return sequences[-1][np.argmax(final_preds)]


def predict_f_for_stateful_rnn(rnn, batch_shape, seed, temperature=1.0):
    x = np.zeros((batch_shape))
    assert isinstance(seed, (list, tuple))
    preds_array = []
    for batch_begin in range(0, len(seed), batch_shape[0]):
        for i, s in enumerate(seed[batch_begin:batch_begin+batch_shape[0]]):
            x[i] = s
        preds = rnn.predict(x, verbose=0)[:i+1, -1, :]
        preds = np.log(preds + 1e-10) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds, axis=-1)[:, None]
        preds_array += list(map(np.squeeze, np.vsplit(preds, i+1)))
    return preds_array