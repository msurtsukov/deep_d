from nltk.tokenize import RegexpTokenizer
import pymorphy2 as pm
import gensim
from scipy.spatial.distance import cosine
import numpy as np


def get_morph():
    return pm.MorphAnalyzer()


def get_tokenizer():
    return RegexpTokenizer(r'\w+')


def get_w2v(load_dir):
    path = load_dir + 'w2v_model.w2v'
    try:
        f = open(path)
        f.close()
    except Exception:
        raise Exception('No file ' + path)
    return gensim.models.Word2Vec.load(path)


def tokenize(sentence, tokenizer):
    return tokenizer.tokenize(sentence)


def normalize(words, morph):
    return [morph.parse(word)[0].normal_form for word in words]


def tokenize_normalize(sentence, tokenizer, morph):
    return normalize(tokenize(sentence, tokenizer), morph)


def sentence_to_vec(sentence, tokenizer, morph, w2v_model, coef='tfidf'):
    words = [word for word in tokenize_normalize(sentence, tokenizer, morph) if word in w2v_model.wv.vocab.keys()]
    vecs = [w2v_model[word] for word in words]
    if len(vecs) == 0:
        return np.zeros(w2v_model.vector_size) + 0.00001
    if coef == 'uniform':
        return sum(vecs) / len(vecs)
    elif coef == 'tfidf':
        coef = np.array([1./np.log(1./w2v_model.wv.vocab[word].count) for word in words])
        coef /= sum(coef)
        return np.dot(coef, np.array(vecs))
    else:
        raise Exception('coef may be "uniform" or "tfidf" but not ' + str(coef))


def load(w2v_dir):
    return get_tokenizer(), get_morph(), get_w2v(w2v_dir)


def sentence_similarity(sentence, candidates, tokenizer, morph, w2v_model, coef='tfidf'):
    vec = sentence_to_vec(sentence, tokenizer, morph, w2v_model, coef=coef)
    return np.array([1 - cosine(vec, sentence_to_vec(c, tokenizer, morph, w2v_model)) for c in candidates])
