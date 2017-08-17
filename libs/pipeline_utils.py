import numpy as np
import tensorflow as tf
from libs.utils import load_model, filter_sequence, pad, top_best


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
           batch_size=64, max_iter=1000):
    """Samples n_items phrases which pass filter_sequence"""
    with enc_graph.as_default():
        states = enc_model.calculate_states(enc_session, transformer, phrases=[seed_phrase])
    batch_states = [np.vstack([state]*batch_size) for state in states]
    sampled = []
    with dec_graph.as_default():
        for i in range(max_iter):
            sequences = dec_model.loop_sample(dec_session, transformer, batch_states)
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


def update_probability(list_of_pp_tuples, probability_f):
    phrases, probs = list(zip(*list_of_pp_tuples))  # zip(*...) is inverse to zip(...)
    conditional_probs = probability_f(phrases)
    out_list_of_pp_tuples = list(zip(phrases, np.multiply(probs, conditional_probs)))
    return out_list_of_pp_tuples


def simple_probability_pipeline(seed_phrase, sample_f, dis_fs, topn=1., last_step_only=True):
    sampled = sample_f(seed_phrase=seed_phrase)
    pp_list = wrap_list_with_score(sampled)
    for d_f in dis_fs:
        pp_list = update_probability(pp_list, d_f)  # apply next discrim
        if not last_step_only:
            pp_list = top_best(pp_list, topn)  # select top n probable
    if last_step_only:
        pp_list = top_best(pp_list, topn)  # select top n probable
    return pp_list
