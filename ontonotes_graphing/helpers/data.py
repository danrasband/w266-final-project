# This file helps with splitting the data.

import os
import numpy as np
import dill as pickle
import pandas as pd
from tqdm import tqdm

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

def entity_id_from_row(row):
    return make_entity_id(
        row.document_id,
        row.sentence_index,
        row.start_word_index,
        row.end_word_index,
    )

def sentence_id_from_row(row):
    return '{}:{}'.format(row.document_id, row.sentence_index)

def get_entity_ids():
    with open(_data_path('entity_ids.pkl'), 'rb') as file:
        entity_ids = pickle.load(file)
    return entity_ids

def get_labeled_data():
    entities = pd.read_csv(_data_path('name_entity.csv'))
    entities['entity_id'] = [entity_id_from_row(row) for _, row in entities.iterrows()]
    entities['sentence_id'] = [sentence_id_from_row(row) for _, row in entities.iterrows()]
    return entities.set_index('entity_id')

def get_sentences():
    sentences = pd.read_csv(_data_path('sentence.csv'))
    sentences['sentence_id'] = [sentence_id_from_row(s) for _, s in sentences.iterrows()]
    return sentences.set_index('sentence_id')

def parse_text(nlp, text):
    try:
        return nlp(text)
    except:
        return None

def parse_sentences(sentences, nlp=None):
    n = len(sentences)
    return [parse_text(nlp, sentence.sentence) for _, sentence in tqdm(sentences.iterrows(), total=n)]

def get_documents():
    documents = pd.read_csv('../data/document.csv')
    return documents.set_index('document_id')

def parse_documents(documents, nlp=None):
    n = len(documents)
    return [parse_text(nlp, document.document) for document in tqdm(documents.iterrows(), total=n)]

def get_inputs_from_sentences(sentences):
    '''Creates a DataFrame with tagged entities using spaCy-parsed sentences.
    The `sentences` DataFrame must have the following columns:

        - document_id
        - sentence_index
        - spacy_parsed
    '''
    rows = []
    for _, sentence in sentences.iterrows():
        if sentence.spacy_parsed is not None:
            for ent in sentence.spacy_parsed.ents:
                entity_id = make_entity_id(*[
                    sentence.document_id,
                    sentence.sentence_index,
                    ent.start,
                    ent.end - 1,
                ])
                row = [
                    entity_id,
                    sentence.document_id,
                    sentence_id_from_row(sentence),
                    ent.label_,
                    str(sentence.sentence_index),
                    str(ent.start),
                    str(ent.end - 1),
                    str(ent),
                ]
                rows.append(row)
        else:
            print('Error with {}'.format(sentence))
    
    columns = (
        'entity_id',
        'document_id',
        'sentence_id',
        'type',
        'sentence_index',
        'start_index',
        'end_index',
        'string',
    )
    return pd.DataFrame(data=rows, columns=columns).set_index('entity_id')

def _data_folder():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path + '/../../data'

def _data_path(filename):
    return '{}/{}'.format(_data_folder(), filename)
