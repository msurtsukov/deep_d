import tensorflow as tf
from functools import partial
from libs.pipeline_utils import build_sampler_env, sample
from libs.pipeline_utils import simple_probability_pipeline, simple_mean_probability_pipeline
from libs.pipeline_utils import probability, meaning_probability, filter_phrases_by_words_count
from libs.utils import load_dictionary, load_transformer, load_dis, text_preprocess
import os
import json


def clear():
    os.system('cls')


def main():
    n_items = 10000
    batch_size = 1024
    enc_seq_len = 64
    dec_seq_len = 128
    softmax_temp = 0.7

    with tf.device("/cpu:0"):
        print("Loading believability discriminator")
        dis_be = load_dis('./models/', 'believability')
        dis_be_w = load_dis('./models', 'believability', file_name="discriminator_believability_word_rnn_model_2.h5")
        print("Loading style discriminator")
        dis_st = load_dis('./models/', 'style')

    print("Loading meaning discriminator")
    dis_me = load_dis('./models/', 'meaning')

    print("Loading sampler")
    load_dir = "./models/shm_c3/"
    sampler_args = build_sampler_env(load_dir, batch_size, enc_seq_len, dec_seq_len)

    dictionary = load_dictionary("./data/")
    transformer = load_transformer(load_dir)
    transformer_w = load_transformer("./models", "tok2id_w.pkl")
    exit_cond = False
    while not exit_cond:
        clear()
        with open("sample_params.json", "r", encoding="utf-8") as f:
            args = json.load(f)
        n_items = args["n"]
        batch_size = args["batch_size"]
        softmax_temp = args["softmax_t"]
        coefs = args["coefs"]
        mode = args["mode"]
        min_words = args["min_words"]

        print("Ctrl-C to exit")
        seed_phrase = input("Enter seed phrase\n").lower()
        if seed_phrase != text_preprocess(seed_phrase):
            print('Valid symbols: а-я:!?,."- \\n')
        else:
            size = int(input("Enter sample size\n"))
            me_f = partial(meaning_probability, model=dis_me, seed_phrase=seed_phrase)
            fw_f = partial(filter_phrases_by_words_count, min_count=min_words)
            be_f = partial(probability, model=dis_be, transformer=transformer)
            be_w_f = partial(probability, model=dis_be_w, transformer=transformer_w, extract_words=True, pad_to_len=40)
            st_f = partial(probability, model=dis_st, transformer=transformer)
            sample_f = partial(sample, *sampler_args,
                               dictionary=dictionary,
                               transformer=transformer,
                               n_items=n_items,
                               batch_size=batch_size,
                               softmax_t=softmax_temp)
            print("Sampling...")
            if mode == 'mean':
                sampled = simple_mean_probability_pipeline(seed_phrase, sample_f, [be_f, be_w_f, fw_f, st_f, me_f],
                                                           topn=size, coefs=coefs)
            else:
                sampled = simple_probability_pipeline(seed_phrase, sample_f, [be_f, be_w_f, fw_f, st_f, me_f],
                                                      topn=size)
            print("Sampled, score")
            for sent, score in sampled:
                print(sent, score)
            input("Hit enter to continue\n")


if __name__ == "__main__":
    main()
