from __future__ import print_function
import tensorflow as tf

import argparse
import os
from six.moves import cPickle
from libs.utils import load_model

from six import text_type


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save/shm_c1',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default="а также поставками из Российской Федерации, проходящих \
по линиям, которые исключены из перечня межгосударственных, перетоки по ним не ведутся в общем сальдо \
Украина-Россия-Беларусь, - написал Ковальчук в своем фейсбуке. \nПоставки на неконтролируемые Киевом территории \
Луганской области были приостановлены в ночь на 25 апреля из-за больших долгов.\n",
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')
    parser.add_argument('--n_samples', type=int, default=15,
                        help='number of samples')

    args = parser.parse_args()
    sample(args)

# todo change (chars, vocab) -> transformer
def sample(args):
    with tf.Session() as sess:
        model, chars, vocab = load_model(args.save_dir, sess, training=False)
        for i in range(args.n_samples):
            print('sample ', i)
            print(model.sample(sess, chars, vocab, args.n, args.prime, args.sample)) # .encode('utf-8'))

if __name__ == '__main__':
    main()
