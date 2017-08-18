from __future__ import print_function
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle
from libs.utils import TextLoader
from libs.model import Model


def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/checkov',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='models/shm_c2',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='shm',
                        help='rnn, gru, lstm, shm, or nas')
    parser.add_argument('--boundary_symbols', type=str, nargs='*', default=[' ', '\n'],
                        help='default (only shm): " ", "\\n"')
    parser.add_argument('--layer_norm', type=int, default=1,
                        help='whether to use layer normalization (only shm)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=64,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.9,
                        help='decay rate for rmsprop')
    parser.add_argument('--decay_every', type=int, default=1000,
                        help='decay every n-th batch (0 for decay between epochs)')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')
    parser.add_argument('--init_from', type=str, default='models/shm_c1',
                        help="""continue training from saved model at this path.
Path must contain files saved by previous training process:
'config.pkl'        : configuration;
'transformer.pkl'   : vocabulary definitions;
'checkpoint'        : paths to model file(s) (created by tf).
Note: this file contains absolute paths, be careful when moving files around;
'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)


def train(args):
    _shm = args.model == 'shm'
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, shm=_shm)
    args.vocab_size = data_loader.transformer.vocab_size
    args.boundary_symbols = [data_loader.transformer.vocab[sym] for sym in args.boundary_symbols]

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from), " %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, "config.pkl")),\
            "config.pkl file does not exist in path %s" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, "transformer.pkl")),\
            "transformer.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme] == vars(args)[checkme],\
                "Command line argument and saved model disagree on '%s' " % checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'transformer.pkl'), 'rb') as f:
            transformer = cPickle.load(f)
        assert transformer.saved_chars == data_loader.transformer.tokens, \
            "Data and loaded model disagree on character set!"
        assert transformer.saved_vocab == data_loader.transformer.vocab, \
            "Data and loaded model disagree on dictionary mappings!"

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'transformer.pkl'), 'wb') as f:
        cPickle.dump(data_loader.transformer, f)

    model = Model(args)

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        lr = 0
        for e in range(args.num_epochs):
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for b in range(data_loader.num_batches):
                start = time.time()
                if _shm:
                    x, y, zs = data_loader.next_batch()
                else:
                    x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                if args.decay_every:
                    new_lr = args.learning_rate * (args.decay_rate ** (b // args.decay_every))
                else:
                    new_lr = args.learning_rate * (args.decay_rate ** e)
                if new_lr != lr:
                    lr = new_lr
                    sess.run(tf.assign(model.lr, lr))
                if _shm:
                    for z, placeholder in zip(zs, model.boundary_data):
                        feed[placeholder] = z
                if _shm or args.model == 'gru':
                    for i, h in enumerate(model.initial_state):
                        feed[h] = state[i]
                else:
                    for i, (c, h) in enumerate(model.initial_state):
                        feed[c] = state[i].c
                        feed[h] = state[i].h
                # instrument for tensorboard
                start_run = time.time()
                # _ = sess.run([model.train_op], feed)
                summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                end_run = time.time()
                writer.add_summary(summ, e * data_loader.num_batches + b)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}, time/run = {:.3f}"
                      .format(e * data_loader.num_batches + b,
                              args.num_epochs * data_loader.num_batches,
                              e, train_loss, end - start, end_run - start_run))
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                        or (e == args.num_epochs - 1 and
                            b == data_loader.num_batches - 1):
                    # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
