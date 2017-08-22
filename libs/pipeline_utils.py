import numpy as np
import tensorflow as tf
from libs.utils import load_model, filter_sequence, pad, top_best
from libs.meaning_discrimination import sentence_similarity

def build_sampler_env(load_dir, batch_size=64, enc_seq_len=64, dec_seq_len=201):
    enc_g = tf.Graph()
    with enc_g.as_default():
        with tf.device("/cpu:0"):
            enc_sess = tf.Session()
            enc_model = load_model(load_dir, enc_sess, False, decoding=False, seq_length=enc_seq_len, batch_size=batch_size)

    dec_g = tf.Graph()
    with dec_g.as_default():
        dec_sess = tf.Session()
        dec_model = load_model(load_dir, dec_sess, False, decoding=True, seq_length=dec_seq_len, batch_size=batch_size)
    return enc_model, enc_sess, enc_g, dec_model, dec_sess, dec_g


def sample(enc_model, enc_session, enc_graph, dec_model, dec_session, dec_graph,
           dictionary, transformer, seed_phrase, n_items,
           batch_size=64, max_iter=1000, softmax_t=1.0):
    """Samples n_items phrases which pass filter_sequence"""
    with enc_graph.as_default():
        states = enc_model.calculate_states(enc_session, transformer, phrases=[seed_phrase])
    batch_states = [np.vstack([state]*batch_size) for state in states]
    sampled = []
    with dec_graph.as_default():
        for i in range(max_iter):
            sequences = dec_model.loop_sample(dec_session, transformer, batch_states, softmax_t=softmax_t)
            for seq in sequences:
                sampled += filter_sequence(seq, dictionary=dictionary)
            if len(sampled) >= n_items:
                break
        sampled = sampled[:n_items]
    return sampled


def wrap_list_with_score(phrases_list, value=1.):
    """For now is only needed to give each sampled phrase basic probability of 1"""
    return [tuple((p, value)) for p in phrases_list]


def probability(phrases_list, model, transformer):
    """Returns probability of each phrase under given discriminator"""

    pad_idx = len(transformer.tokens)  # pad with new element
    X = np.array(
            [pad(transformer.transform(p), to_len=200, with_what=pad_idx) for p in phrases_list]
        )

    preds = model.predict(X)
    return preds[:, 0]


def meaning_probability(phrases_list, model, seed_phrase, coef="uniform"):
    sim = sentence_similarity(seed_phrase, phrases_list, model.tokenizer, model.morph, model.w2v, coef=coef)
    prob = (sim + 1.) / 2.
    return prob


def update_probability(list_of_pp_tuples, probability_f, mode='prob', coef=1.0):
    phrases, probs = list(zip(*list_of_pp_tuples))  # zip(*...) is inverse of zip(...)
    conditional_probs = coef*probability_f(phrases)
    if mode == 'sum':
        out_list_of_pp_tuples = list(zip(phrases, np.add(probs, conditional_probs)))
    else:
        out_list_of_pp_tuples = list(zip(phrases, np.multiply(probs, conditional_probs)))
    return out_list_of_pp_tuples


def weighted_probabilities_sum(list_of_lists_of_pp_tuples, coefs='uniform'):
    if type(coefs) == str:
        if coefs == 'uniform':
            coefs = np.array([1.] * len(list_of_lists_of_pp_tuples), dtype='float32')
            coefs /= sum(coefs)
        else:
            raise Exception('coefs must be "uniform" or an array of numbers')
    coefs = np.array(coefs, dtype='float32')
    if sum(coefs) < 1e-7:
        raise Exception('sum(coefs) must be not zero')
    coefs /= sum(coefs)
    if len(coefs.shape) != 1:
        raise Exception('coefs must be a 1-dimensional float array')
    if len(coefs) != len(list_of_lists_of_pp_tuples):
        raise Exception('length of coefs must be equal to the length of list_of_lists_of_pp_tuples')
    if len(list_of_lists_of_pp_tuples) == 0:
        raise Exception('length of list_of_lists_of_pp_tuples must be more than 0')
    probs = []
    for list_of_pp_tuples in list_of_lists_of_pp_tuples:
        phrases, cur_probs = list(zip(*list_of_pp_tuples))
        probs.append(cur_probs)
    probs = np.array(probs, dtype='float32')
    probs = np.dot(coefs, probs)
    return list(zip(phrases, probs))


def simple_probability_pipeline(seed_phrase, sample_f, dis_fs, topn=1., last_step_only=True):
    sampled = sample_f(seed_phrase=seed_phrase)
    pp_list = wrap_list_with_score(sampled)
    for d_f in dis_fs:
        pp_list = update_probability(pp_list, d_f)  # apply next discriminator
        if not last_step_only:
            pp_list = top_best(pp_list, topn)  # select top n probable
    if last_step_only:
        pp_list = top_best(pp_list, topn)  # select top n probable
    return pp_list


def simple_mean_probability_pipeline(seed_phrase, sample_f, dis_fs, topn=0.002, coefs='uniform'):
    if type(coefs) == str:
        if coefs == 'uniform':
            coefs = np.array([1.] * len(dis_fs), dtype='float32')
        else:
            raise Exception('coefs must be "uniform" or an array of numbers')
    coefs = np.array(coefs, dtype='float32')
    coefs /= sum(coefs)

    sampled = sample_f(seed_phrase=seed_phrase)
    pp_list = wrap_list_with_score(sampled)
    for c, d_f in zip(coefs, dis_fs):
        pp_list = update_probability(pp_list, d_f, mode='sum', coef=c)  # apply next discriminator

    pp_list = top_best(pp_list, topn)  # select top n probable
    return pp_list
