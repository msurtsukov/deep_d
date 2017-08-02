from __future__ import print_function
import tensorflow as tf
import numpy as np

import argparse
import os
from six.moves import cPickle
from six import text_type
from libs.utils import load_model


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
    parser.add_argument('--phrase', type=text_type, default=u'просто очередная \n тестовая фраза',
                        help='test phrase')

    args = parser.parse_args()
    test_hidden(args)


def test_hidden(args):
    with tf.Session() as sess:
        model, chars, vocab = load_model(args.save_dir, sess, training=False)

        hidden_norms = model.get_hidden_l2_norm(sess, vocab, args.phrase)
        for h_norm, char in zip(hidden_norms, args.phrase):
            print(np.squeeze(h_norm), char)

if __name__ == '__main__':
    main()
