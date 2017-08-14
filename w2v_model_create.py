import gensim


with open('./data/normed_text.txt', 'r', encoding='utf-8') as inp:
    corpora = inp.read()
corpora = corpora.replace('<@>_<@>_<@>', '<@@@>')
corpora = corpora.replace('<@>_<@>', '<@@>')
sentences = [[word for word in sent.split('_') if word != ''] for sent in corpora.split('\n')]
w2v = gensim.models.Word2Vec(sentences, size=200, min_count=0)
w2v.save('./models/w2v_model.w2v')
