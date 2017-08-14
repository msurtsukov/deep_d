import numpy as np
from libs.utils import filter_sequence


def sample(enc_model, enc_session, enc_graph, dec_model, dec_session, dec_graph,
           dictionary, transformer, seed_phrase, n_items,
           batch_size=64, max_iter=1000):
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