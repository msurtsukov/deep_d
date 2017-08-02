import numpy as np
from libs.utils import filter_sequence


def sample(enc_model, enc_session, dec_model, dec_session,
           dictionary, transformer, seed_phrase, n_items,
           batch_size=64, max_iter=100):
    state = enc_model.calculate_states(enc_session, transformer, phrases=[seed_phrase])[0]
    state = np.vstack([state]*batch_size)
    sampled = []
    for i in range(max_iter):
        sequences = dec_model.loop_sample(dec_session, transformer, state)
        for seq in sequences:
            sampled += filter_sequence(seq, dictionary=dictionary)
        if len(sampled) >= n_items:
            break
    sampled = sampled[:n_items]
    return sampled
