import tensorflow as tf
from libs.utils import load_model, load_transformer, check_sent, filter_sequence
from collections import defaultdict
import json


file_name = './data/dostoewskij/input.txt'
save_to = './data/dostoewskij/sampled.json'
model_path = './save/shm_c1'
loops = 8

with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

sentences = text.split('\n')
batch_size = 325
batches = []
for i in range(len(sentences)//batch_size):
    batch = sentences[i*batch_size:(i+1)*batch_size]
    batches.append(batch)

sess = tf.Session()
model = load_model(model_path, sess, training=False, decoding=False, seq_length=64, batch_size=batch_size)
transformer = load_transformer(model_path)

states = []
print('calculating states')
for i, batch in enumerate(batches):
    states.append(model.calculate_states(sess, transformer, phrases=batch))
    print(i+1, '/', len(batches),  end='\r')

with open('./data/words_dictionary.txt', 'r', encoding='utf-8') as f:
    dictionary = f.read().split('_')

g = tf.Graph()
with g.as_default():
    sess_sample = tf.Session()
    model_sample = load_model(model_path, sess_sample, training=False, decoding=True,
                              seq_length=201, batch_size=batch_size)

print('sampling')
sampled_by_seed = defaultdict(list)
for loop in range(loops):
    print('loop:', loop)
    for i, state, batch in zip(range(len(batches)), states, batches):
        samples = model_sample.loop_sample(sess_sample, transformer, state)
        for sample, seed in zip(samples, batch):
            if check_sent(seed):
                sampled_by_seed[seed] += filter_sequence(sample, dictionary=dictionary)
        print(i+1, '/', len(batches),  end='\r')

with open(save_to, 'w', encoding='utf-8') as f:
    json.dump(sampled_by_seed, f, ensure_ascii=False)
