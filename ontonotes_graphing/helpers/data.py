# This file helps with splitting the data.

import numpy as np

def train_dev_test_split_ids(ids, train_portion=0.7, dev_portion=0.2, seed=None):
    '''Built out three lists of ids for train, dev, and test portions.'''
    # Clone the ids list.
    ids = ids.copy()
    
    # Set a random seed if one is provided.
    if seed: np.random.seed(seed)
    # Shuffle the ids!
    np.random.shuffle(ids)
    
    # Split the ids according to the specified portions.
    n = len(ids)
    n_train = int(n * train_portion)
    dev_end = n_train + int(n * dev_portion)
    
    return [ids[0:n_train], ids[n_train:dev_end], ids[dev_end:]]


def make_entity_id(document_id, sentence_index, start_word_index, end_word_index):
    return ':'.join([
        document_id,
        str(sentence_index),
        str(start_word_index),
        str(end_word_index),
    ])
