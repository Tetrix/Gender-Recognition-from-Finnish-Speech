import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np
import pickle
import gensim
import fasttext
import math

import utils.prepare_data as prepare_data
from model import Encoder, Transformer
from config.config import *
from train import train
from get_predictions import get_predictions


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# load features and labels
print('Loading data...')



# Lahjoita Puhetta data
#features_train = prepare_data.load_features_combined('data/gender_modeling/features/train.npy', max_len)
#trn_train, topic_train = prepare_data.load_transcripts('data/gender_modeling/transcripts_gender/train.txt')
#
#features_dev = prepare_data.load_features_combined('data/gender_modeling/features/dev.npy', max_len)
#trn_dev, topic_dev = prepare_data.load_transcripts('data/gender_modeling/transcripts_gender/dev.txt')


# test Lahjoita Puhetta
features_train = prepare_data.load_features_combined('data/gender_modeling/features/test.npy', max_len)
trn_train, topic_train = prepare_data.load_transcripts('data/gender_modeling/transcripts_gender/test.txt')


features_train = features_train
trn_train = trn_train
topic_train = topic_train


features_dev = features_train
trn_dev = trn_train
topic_dev = topic_train


print('Done...')


print('Loading embeddings...')
embeddings = fasttext.load_model('weights/embeddings/cc.fi.300.bin')
print('Done...')



# generate index dictionaries
#char2idx, idx2char = prepare_data.encode_data(target_train)


#topic2idx = {}
#for i in topic_train:
#    i = i[0]
#    if i not in topic2idx.keys():
#        topic2idx[i] = len(topic2idx) + 1
#
#idx2topic = {v: k for k, v in topic2idx.items()}

# generate index dictionaries
#with open('weights/topic2idx.pkl', 'wb') as f:
#    pickle.dump(topic2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
#
#with open('weights/idx2topic.pkl', 'wb') as f:
#    pickle.dump(idx2topic, f, protocol=pickle.HIGHEST_PROTOCOL)



# For topic detection
#with open('weights/topic2idx.pkl', 'rb') as f:
#    topic2idx = pickle.load(f)
#with open('weights/idx2topic.pkl', 'rb') as f:
#    idx2topic = pickle.load(f)


# for gender detection
topic2idx = {'Mies': 0, 'Nainen': 1}
idx2topic = {0: 'Mies', 1: 'Nainen'}


# for age detection
#topic2idx = {'1-10': 1, '11-20': 2, '21-30': 3, '31-40': 4, '41-50': 5, '51-60': 6, '61-70': 7, '71-80': 8, '81-90': 9, '91-100': 10, '101+': 11}
#topic2idx = {'1-20': 0, '21-60': 1, '61-101+': 2}
#idx2topic = {v: k for k, v in topic2idx.items()}



with open('weights/char2idx.pkl', 'rb') as f:
    char2idx = pickle.load(f)
with open('weights/idx2char.pkl', 'rb') as f:
    idx2char = pickle.load(f)



# convert topics to indices
indexed_topic_train = prepare_data.topic_to_idx(topic_train, topic2idx)
indexed_topic_dev = prepare_data.topic_to_idx(topic_dev, topic2idx)


# convert words to vectors
indexed_word_train = prepare_data.word_to_idx(trn_train, embeddings)
indexed_word_dev = prepare_data.word_to_idx(trn_dev, embeddings)


# combine features and topics in a tuple
train_data = prepare_data.combine_data(features_train, indexed_topic_train, indexed_word_train)
dev_data = prepare_data.combine_data(features_dev, indexed_topic_dev, indexed_word_dev)


# remove extra data that doesn't fit in batch
train_data = prepare_data.remove_extra(train_data, batch_size)
dev_data = prepare_data.remove_extra(dev_data, batch_size)


class_1 = 0
class_2 = 0
class_3 = 0

indexed_topic_train = indexed_topic_train[:3520]

for sample in indexed_topic_train:
    if sample.item() == 0:
        class_1 += 1
    if sample.item() == 1:
        class_2 += 1
    if sample.item() == 2:
        class_3 += 1


class_sample_counts = [class_1, class_2, class_3]
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
sample_weights = np.array([weights[t.item() - 1] for t in indexed_topic_train])
sample_weights = torch.from_numpy(sample_weights)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


pairs_batch_train = DataLoader(dataset=train_data,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)

pairs_batch_dev = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)




transformer = Transformer(features_train[0].size(1), len(topic2idx), n_head_encoder, d_model, n_layers_encoder, max_len).to(device)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)


total_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print('The number of trainable parameters is: %d' % (total_trainable_params))


# train
if skip_training == False:
    print('Training...')
    #weight_1 = len(indexed_topic_train) / (3 * class_1)
    #weight_2 = len(indexed_topic_train) / (3 * class_2)
    #weight_3 = len(indexed_topic_train) / (3 * class_3)
    
    #weight_1 = 1 / class_1
    #weight_2 = 1 / class_2
    #weight_3 = 1 / class_3

    #weight = torch.Tensor([weight_1, weight_2, weight_3]).to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr) 

    #checkpoint = torch.load('weights/gender_new/state_dict_26.pt', map_location=torch.device('cpu'))
    #transformer.load_state_dict(checkpoint['transformer'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

    train(pairs_batch_train, 
            pairs_batch_dev,
            transformer,
            criterion,
            optimizer,
            num_epochs,
            batch_size,
            len(features_train),
            len(features_dev),
            device) 
else:
    checkpoint = torch.load('weights/gender/state_dict_34.pt', map_location=torch.device('cpu'))
    transformer.load_state_dict(checkpoint['transformer'])


batch_size = 1

pairs_batch_train = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)


get_predictions(transformer, batch_size, idx2topic, pairs_batch_train)
