import os
import sys
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from itertools import product
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from scipy.sparse import vstack, bmat, diags

if len(sys.argv) != 2:
    sys.exit("Use: python build_graph.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
# build corpus
dataset = sys.argv[1]

if dataset not in datasets:
    sys.exit("wrong dataset name")

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
#_, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])

word_embeddings_dim = 300
word_vector_map = {}

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

print('Reading data...')

f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()
# print(doc_train_list)
# print(doc_test_list)

doc_content_list = []
f = open('data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()
# print(doc_content_list)

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
#print(len(train_ids))
random.shuffle(train_ids)

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
#print(len(test_ids))
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids
#print(ids)
#print(len(ids))
#print(ids[:10])


shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[id])
    shuffle_doc_words_list.append(doc_content_list[id].split())
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(doc_content_list[id] for id in ids)

f = open('data/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()

print('Embedding...')
# build vocab
word_freq = {}
word_set = set()
#for doc_words in shuffle_doc_words_list:
for words in shuffle_doc_words_list:
    #words = doc_words.split()
    word_set.update(words)
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = sorted(list(word_set))
vocab_size = len(vocab)

word_doc_list = {}
word_doc_rep_list = {}

for i in range(len(shuffle_doc_words_list)):
    #doc_words = shuffle_doc_words_list[i]
    #words = doc_words.split()
    words = shuffle_doc_words_list[i]
    appeared = set()
    for word in words:
        if word not in word_doc_list:
            word_doc_list[word] = list()
            word_doc_rep_list[word] = list()
        if word not in appeared:
            word_doc_list[word].append(i)
            appeared.add(word)
        word_doc_rep_list[word].append(i)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
f.write(vocab_str)
f.close()

'''
Word definitions begin
'''
#print('Word definitions...')
'''
definitions = []
print(vocab[:10])
for word in tqdm(vocab):
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)
print(string[:1000])

f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)

f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()
'''


word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])
#print(word_embeddings_dim)

# word embedding matrix
wm = np.matrix(embd)


'''
Word definitions end
'''

# label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = sorted(list(label_set))

#label_list_str = '\n'.join(label_list)
#f = open('data/corpus/' + dataset + '_labels.txt', 'w')
#f.write(label_list_str)
#f.close()

print('Feature vectors...')

#print('Building feature vectors...')
# x: feature vectors of training docs, no initial features
# slect 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
# different training rates

#real_train_doc_names = shuffle_doc_name_list[:real_train_size]
#real_train_doc_names_str = '\n'.join(real_train_doc_names)

#f = open('data/' + dataset + '.real_train.name', 'w')
#f.write(real_train_doc_names_str)
#f.close()

# train document word matrix
rows = []
cols = []
data = []
for i in range(real_train_size):
    wids = [word_id_map[word] for word in shuffle_doc_words_list[i]]
    rows.extend(i for _ in range(len(wids)))
    cols.extend(wids)
    data.extend(1.0/len(wids) for _ in range(len(wids)))
x = sp.csr_matrix((data, (rows, cols)), shape=(real_train_size, vocab_size))
x = x * wm
#print(x.shape)
#print(x)


#print('Building labels...')
y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
#print(y)

# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

# test document word matrix
rows = []
cols = []
data = []
for i in range(test_size):
    wids = [word_id_map[word] for word in shuffle_doc_words_list[i + train_size]]
    rows.extend(i for _ in range(len(wids)))
    cols.extend(wids)
    data.extend(1.0/len(wids) for _ in range(len(wids)))
tx = sp.csr_matrix((data, (rows, cols)), shape=(test_size, vocab_size))
tx = tx * wm
#print(tx.shape)
#print(tx)


ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
#print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

rows = []
cols = []
data = []
for i in range(train_size):
    wids = [word_id_map[word] for word in shuffle_doc_words_list[i]]
    rows.extend(i for _ in range(len(wids)))
    cols.extend(wids)
    data.extend(1.0/len(wids) for _ in range(len(wids)))
allx = sp.csr_matrix((data, (rows, cols)), shape=(train_size, vocab_size))
allx = allx * wm
allx = vstack([allx, wm])
#print(allx.shape)
#print(allx)


ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

#print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20

print('PMIs...')


rows = []
cols = []
data = []
window_counter = 0
for i in range(len(shuffle_doc_words_list)):
    length = len(shuffle_doc_words_list[i])
    size = min(length, window_size)
    for j in range(length - size + 1):
        unique = list(set(shuffle_doc_words_list[i][j:j+size]))
        rows.extend(window_counter for _ in range(len(unique)))
        cols.extend(word_id_map[w] for w in unique)
        data.extend(1 for _ in range(len(unique)))
        window_counter += 1
wfm = sp.coo_matrix((data, (rows, cols)), shape=(window_counter, vocab_size))
wfm = wfm.transpose() * wfm
wfm = wfm.diagonal()



rows = []
cols = []
data = []
window_counter = 0
for i in range(len(shuffle_doc_words_list)):
    length = len(shuffle_doc_words_list[i])
    size = min(length, window_size)
    for j in range(length - size + 1):
        unique = shuffle_doc_words_list[i][j:j+size]
        #unique = list(set(shuffle_doc_words_list[i][j:j+size]))   for this part let's count the duplicates!
        rows.extend(window_counter for _ in range(len(unique)))
        cols.extend(word_id_map[w] for w in unique)
        data.extend(1 for _ in range(len(unique)))
        window_counter += 1
wpm = sp.coo_matrix((data, (rows, cols)), shape=(window_counter, vocab_size))
wpm = wpm.transpose() * wpm
wpm = wpm - diags(wpm.diagonal())



row = []
col = []
weight = []

# pmi as weights
pmi = wpm.copy()
rows, cols = pmi.nonzero()
pmi.data = np.clip(np.log(np.divide(pmi.data, wfm[rows] * wfm[cols] / float(window_counter))), 0, None)


print('Adjacency matrix...')


# word vector cosine similarity as weights

'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''


wdf = np.array([word_doc_freq[vocab[j]] for j in range(vocab_size)])
idf = np.log(float(len(shuffle_doc_words_list)) / wdf)

rows = []
cols = []
for doc_id in range(len(shuffle_doc_words_list)):
    words = shuffle_doc_words_list[doc_id]
    wids = [word_id_map[word] for word in words]
    rows.extend(doc_id for _ in range(len(wids)))
    cols.extend(wids)
data = idf[cols]
tfidf = sp.csr_matrix((data, (rows, cols)), shape=(len(shuffle_doc_words_list), vocab_size))
#print(tfidf.shape)
#print(tfidf)


node_size = train_size + vocab_size + test_size

adj = bmat([
    [None, tfidf[:train_size], None],
    [sp.csr_matrix((vocab_size, train_size)), pmi, sp.csr_matrix((vocab_size, test_size))],
    [None, tfidf[train_size:], None]
])


# dump objects
f = open("data/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()
