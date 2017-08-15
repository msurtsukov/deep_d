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
    parser.add_argument('--save_dir', type=str, default='models/shm_c1',
                        help='model directory to store checkpointed models')
    parser.add_argument('--stop_sym', type=str, default=' ',
                        help='symbol to sample until')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
    parser.add_argument('--phrase', type=text_type, default="После этого Россия приняла решение начать поставку электроэнергии в обесточенный регион.",
                        help='phrase to rebuild')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')
    parser.add_argument('--prime', type=text_type,
                        default="а также поставками из Российской Федерации, проходящих \
по линиям, которые исключены из перечня межгосударственных, перетоки по ним не ведутся в общем сальдо \
Украина-Россия-Беларусь, - написал Ковальчук в своем фейсбуке. \nПоставки на неконтролируемые Киевом территории \
Луганской области были приостановлены в ночь на 25 апреля из-за больших долгов.\n",
                        help='phrase to rebuild')

    args = parser.parse_args()
    rebuild_phrase(args)


# todo change (chars, vocab) -> transformer
def rebuild_phrase(args):
    with tf.Session() as sess:
        model, chars, vocab = load_model(args.save_dir, sess, training=False)

        words = args.phrase.split(args.stop_sym)
        prime = args.prime + args.stop_sym
        rez = words[0] + args.stop_sym
        for word in words:
            prime = prime + word + args.stop_sym
            new_phrase = model.sample(sess, chars, vocab, args.n, prime, args.sample, args.stop_sym)
            new_word = new_phrase.split(args.stop_sym)[-1]
            rez += new_word + args.stop_sym
        print(prime)
        print()
        print()
        print(rez)
if __name__ == '__main__':
    main()
