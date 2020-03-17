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


def write_list(l, file):
    with open(file, 'w') as f:
        for item in l:
            f.write(str(item))
            f.write('\n')
    return True



if __name__ == '__main__':
    print('Reading data...')
    dataset = '20ng'
    ids, names, docs, labels, train_size, val_size, test_size = read_dataset(dataset)
    # In each array (ids, names, docs, labels) first there are training items
    # then there are validation items and then there are test items, each with
    # size train_size, val_size, test_size, respectively.
