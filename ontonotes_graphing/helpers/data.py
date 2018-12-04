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