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


def read_dataset(ds, shuffle=True):
    names_file = 'data/' + ds + '.txt'
    docs_file = 'data/corpus/' + ds + '.clean.txt'
    names = []
    docs = []
    train_ids = []
    val_ids = []
    test_ids = []
    labels = []
    
    # read names + train ids + test ids
    with open(names_file, 'r') as f:
        for i, l in enumerate(f.readlines()):
            n = l.strip().split('\t')
            names.append(n)
            if n[2].startswith('sci') or n[2].startswith('rec'):
                train_ids.append(i)
            elif n[2].startswith('comp'):
                val_ids.append(i)
            else:
                test_ids.append(i)
    labels = [n[2] for n in names]
    
    # read docs
    with open(docs_file, 'r') as f:
        docs = [l.strip() for l in f.readlines()]
    
    # shuffle
    if shuffle:
        random.shuffle(train_ids)
        random.shuffle(val_ids)
        random.shuffle(test_ids)
    ids = train_ids + val_ids + test_ids
    names = [names[i] for i in ids]
    docs = [docs[i] for i in ids]
    labels = [labels[i] for i in ids]
    train_size = len(train_ids)
    val_size = len(val_ids)
    test_size = len(test_ids)
    return ids, names, docs, labels, train_size, val_size, test_size


def build_vocab(docs_words):
    word_id_map = {}
    vocab = []
    for words in docs_words:
        for w in words:
            if w not in word_id_map:
                word_id_map[w] = len(vocab)
                vocab.append(w)
    return vocab, word_id_map


def build_freq_matrix(docs_wids, word_id_map):
    rows = []
    cols = []
    data = []
    for i, doc in enumerate(docs_wids):
        rows.extend(i for _ in range(len(doc)))
        cols.extend(doc)
        data.extend(1 for _ in range(len(doc)))
    freq = sp.csr_matrix((data, (rows, cols)), shape=(len(docs_wids), len(word_id_map)))
    return freq


def get_tfidf(freq, idf):
    tfidf = freq.copy()
    rows, cols = tfidf.nonzero()
    tfidf.data = np.array(tfidf.data) * idf[cols]
    return tfidf


def get_label_ids(labels):
    label_id_map = {}
    label_list = []
    for label in labels:
        if label not in label_id_map:
            label_id_map[label] = len(label_list)
            label_list.append(label)
    return label_list, label_id_map


# def build_window_freq_matrix(docs_wids, vocab_size, window_size=20):
#     rows = []
#     cols = []
#     data = []
#     window_counter = 0
#     for i, wids in enumerate(docs_wids):
#         length = len(wids)
#         size = min(length, window_size)
#         for j in range(length - size + 1):
#             unique = list(set(wids[j:j+size])) # duplicates not included!
#             rows.extend(window_counter for _ in range(len(unique)))
#             cols.extend(wid for wid in unique)
#             data.extend(1 for _ in range(len(unique)))
#             window_counter += 1
#     wfm = sp.coo_matrix((data, (rows, cols)), shape=(window_counter, vocab_size))
#     wfm = wfm.transpose() * wfm
#     wfm = wfm.diagonal()
#     return wfm, window_counter


# def build_window_cofreq_matrix(docs_wids, vocab_size, window_size=20):
#     rows = []
#     cols = []
#     data = []
#     window_counter = 0
#     for i, wids in enumerate(docs_wids):
#         length = len(wids)
#         size = min(length, window_size)
#         for j in range(length - size + 1):
#             unique = wids[j:j+size] # duplicates included!
#             rows.extend(window_counter for _ in range(len(unique)))
#             cols.extend(wid for wid in unique)
#             data.extend(1 for _ in range(len(unique)))
#             window_counter += 1
#     wpm = sp.coo_matrix((data, (rows, cols)), shape=(window_counter, vocab_size))
#     wpm = wpm.transpose() * wpm
#     wpm = wpm - diags(wpm.diagonal())
#     return wpm, window_counter


def build_window_freqs(docs_wids, vocab_size, window_size=20):
    rows = []
    cols = []
    data = []
    window_counter = 0
    for i, wids in enumerate(docs_wids):
        length = len(wids)
        size = min(length, window_size)
        for j in range(length - size + 1):
            unique = list(set(wids[j:j+size])) # duplicates not included!
            rows.extend(window_counter for _ in range(len(unique)))
            cols.extend(wid for wid in unique)
            data.extend(1 for _ in range(len(unique)))
            window_counter += 1
    wpm = sp.coo_matrix((data, (rows, cols)), shape=(window_counter, vocab_size))
    wpm = wpm.transpose() * wpm
    wfm = wpm.diagonal()
    wpm = wpm - diags(wfm)
    return wfm, wpm, window_counter


def load_or_build_embedding(ds):
    # Read Word Vectors
    # word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
    # word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
    #_, embd, word_vector_map = loadWord2Vec(word_vector_file)
    # word_embeddings_dim = len(embd[0])
    try:
        word_vector_file = 'data/corpus/' + ds + '_word_vectors.txt'
        _, embd, word_vector_map = loadWord2Vec(word_vector_file)
        word_embeddings_dim = len(embd[0])
        #print(word_embeddings_dim)

        # word embedding matrix
        wm = np.matrix(embd)
        return wm
    except:
        definitions = []
        for word in vocab:
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

        #string = '\n'.join(definitions)
        #f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
        #f.write(string)
        #f.close()

        tfidf_vec = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf_vec.fit_transform(definitions)
        tfidf_matrix_array = tfidf_matrix.toarray()
        #print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

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
        f = open('data/corpus/' + ds + '_word_vectors.txt', 'w')
        f.write(string)
        f.close()
        
        return load_or_build_embedding(ds)


def write_list(l, file):
    with open(file, 'w') as f:
        for item in l:
            f.write(str(item))
            f.write('\n')
    return True

def dump_obj(obj, file):
    with open(file, 'wb') as f:
        pkl.dump(obj, f)



if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        sys.exit("Use: python build_graph.py <dataset>")

    datasets = ['20ng']
    dataset = sys.argv[1]

    if dataset not in datasets:
        sys.exit("wrong dataset name")



    print('Reading data...')
    ids, names, docs, labels, train_size, val_size, test_size = read_dataset(dataset)
    train_size += val_size

    write_list(ids[:train_size], 'data/fs.' + dataset + '.train.index')
    write_list(ids[train_size:], 'data/fs.' + dataset + '.test.index')
    write_list(names, 'data/fs.' + dataset + '_shuffle.txt')
    write_list(docs, 'data/corpus/fs.' + dataset + '_shuffle.txt')

    print('Building vocab...')
    docs_words = [doc.split() for doc in docs]
    vocab, word_id_map = build_vocab(docs_words)
    docs_wids = [[word_id_map[w] for w in doc] for doc in docs_words]

    write_list(vocab, 'data/corpus/fs.' + dataset + '_vocab.txt')

    print('Frequencies...')
    freq_mat = build_freq_matrix(docs_wids, word_id_map)

    print('Embedding...')
    word_mat = load_or_build_embedding(dataset)
    word_embeddings_dim = word_mat.shape[1]
    #print(word_embeddings_dim)

    print('Label IDs...')
    label_list, label_id_map = get_label_ids(labels)
    label_ids = [label_id_map[l] for l in labels]
    label_mat = np.eye(len(label_list))

    write_list(label_list, 'data/corpus/fs.' + dataset + '_labels.txt')


    print('Feature vectors...')
    real_train_size = train_size - val_size

    write_list(names[:real_train_size], 'data/fs.' + dataset + '.real_train.name')

    # train
    train_freq = freq_mat[:real_train_size]
    x = (train_freq / train_freq.sum(1)) * word_mat
    y = label_mat[label_ids[:real_train_size],:]

    # test
    test_freq = freq_mat[train_size:]
    tx = (test_freq / test_freq.sum(1)) * word_mat
    ty = label_mat[label_ids[train_size:],:]

    # all (+words)
    train_freq = freq_mat[:train_size]
    allx = (train_freq / train_freq.sum(1)) * word_mat
    #ally = label_mat[label_ids[:train_size],:]
    ally = label_mat[label_ids[:real_train_size],:]
    allx = vstack([allx, word_mat])
    ally = vstack([ally, sp.csr_matrix((val_size + len(vocab), len(label_list)))])

    #print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    print('PMIs...')
    #window_freq, num_windows = build_window_freq_matrix(docs_wids, len(vocab))
    #window_cofreq, num_windows = build_window_cofreq_matrix(docs_wids, len(vocab))
    window_freq, window_cofreq, num_windows = build_window_freqs(docs_wids, len(vocab))

    # pmi as weights
    pmi = window_cofreq.copy()
    rows, cols = pmi.nonzero()
    pmi.data = np.clip(np.log(np.divide(pmi.data, window_freq[rows] * window_freq[cols] / float(num_windows))), 0, None)


    print('Adjacency matrix...')
    app_mat = freq_mat.copy()
    app_mat[app_mat > 0] = 1
    word_freq_arr = np.asarray(app_mat.sum(0))[0]
    idf_arr = np.log(float(len(docs)) / word_freq_arr)
    tfidf_mat = get_tfidf(freq_mat, idf_arr)

    node_size = train_size + len(vocab) + test_size

    adj = bmat([
        [None, tfidf_mat[:train_size], None],
        [tfidf_mat[:train_size].transpose(), pmi, tfidf_mat[train_size:].transpose()],
        [None, tfidf_mat[train_size:], None]
    ])

    # adj = bmat([
    #     [None, tfidf_mat[:train_size], None],
    #     [sp.csr_matrix((len(vocab), train_size)), pmi, sp.csr_matrix((len(vocab), test_size))],
    #     [None, tfidf_mat[train_size:], None]
    # ])


    dump_obj(x, "data/fs.ind." + dataset + ".x")
    dump_obj(y, "data/fs.ind." + dataset + ".y")
    dump_obj(tx, "data/fs.ind." + dataset + ".tx")
    dump_obj(ty, "data/fs.ind." + dataset + ".ty")
    dump_obj(allx, "data/fs.ind." + dataset + ".allx")
    dump_obj(ally, "data/fs.ind." + dataset + ".ally")
    dump_obj(adj, "data/fs.ind." + dataset + ".adj")



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
