import numpy as np
import os

import torch
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence
from nltk import word_tokenize
from random import randrange


# load features combined in one file
def load_features_combined(features_path, max_len):
    feature_array = []
    features = np.load(features_path, allow_pickle=True)
    
    for feature in features:
        feature = feature[:max_len]
        feature_array.append(autograd.Variable(torch.FloatTensor(feature)))

    return feature_array


def load_features(features_path, max_len):
    feature_array = []
    for file in sorted(os.listdir(features_path)):
        feature = np.load(os.path.join(features_path, file))
        feature = feature[:max_len]
        feature_array.append(autograd.Variable(torch.FloatTensor(feature)))

    return feature_array


def load_transcripts(features_path):
    trn_array = []
    topic_array = []
    with open(features_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    for sent in data:
        sent = sent.rstrip()
        trn_array.append([sent.split('\t')[0]])
        topic_array.append([sent.split('\t')[1]])

    return trn_array, topic_array



def load_tags(features_path):
    tag_array = []
    temp = []
    with open(features_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    for tag in data:
        if tag != '\n':
            tag = tag.rstrip()
            temp.append(tag)
        else:
            tag_array.append(temp)
            temp = []
    
    return tag_array



def load_labels(features_path):
    label_array = []
    for file in sorted(os.listdir(features_path)):
        if os.path.isfile(os.path.join(features_path, file)):
            with open(os.path.join(features_path, file), 'r', encoding='utf-8') as f:
                transcript = f.readlines()
                label_array.append(transcript)

    return label_array


#TODO Add <UNK> as a special token. Currently it is processed as characters
def encode_data(labels_data):
    char2idx = {}
    idx2char = {}


    char2idx['<sos>'] = 1
    idx2char[1] = '<sos>'

    char2idx['<eos>'] = 2
    idx2char[2] = '<eos>'

    char2idx['<UNK>'] = 3
    idx2char[3] = '<UNK>'


    for sent in labels_data:
        sentence = sent[0].split()
        for char in sent[0]:
            if char not in char2idx:
                char2idx[char] = len(char2idx) + 1
                idx2char[len(idx2char) + 1] = char

    return char2idx, idx2char



# for English
def label_to_idx(labels, char2idx):
    res = []
    for sent in labels:
        # for English
        #sent = word_tokenize(sent[0])
        #sent = ' '.join(sent)
        #print(sent)
        sent = sent[0].split()
        sent = ' '.join(sent)
        # end English
        temp_sent = []
        # add <sos> token
        temp_sent.append([char2idx['<sos>']])
        temp_sent.append([char2idx[' ']])

        for char in sent:
        # end English
            try:
                temp_sent.append([char2idx[char]])
            except:
                temp_sent.append([char2idx['z']])
        # add <eos> token
        temp_sent.append([char2idx[' ']])
        temp_sent.append([char2idx['<eos>']])

        res.append(torch.LongTensor(temp_sent))
    return res


#def topic_to_idx(topics, topic2idx):
#    res = []
#    temp_res = []
#    for topic in topics:
#        temp_res.append(topic2idx[topic[0]])
#    
#    res = torch.LongTensor(temp_res)
#    return res


# modified the age boundaries, such that: 1-20 = 1; 21-60 = 2; 61 - 101+ = 3 
def topic_to_idx(topics, topic2idx):
    res = []
    temp_res = []
    for topic in topics:
        if topic[0] in ['1-10', '11-20']:
            topic[0] = '1-20'
        if topic[0] in ['21-30', '31-40', '41-50', '51-60']:
            topic[0] = '21-60'
        if topic[0] in ['61-70', '71-80', '81-90', '91-100', '101+']:
            topic[0] = '61-101+'
        
        temp_res.append(topic2idx[topic[0]])
     
    res = torch.LongTensor(temp_res)
    return res



# TOKENIZER ADDED FOR ENGLISH DATASET!!!!
def prepare_word_sequence(seq, embeddings):
    res = []
    # for Finnish
    #seq = seq.split()
    # for English
    #seq = word_tokenize(seq)
    seq = seq.split()
    for w in seq:
        try:
            res.append(embeddings[w])
        except:
            res.append(np.random.normal(scale=0.6, size=(300, )))
    res = autograd.Variable(torch.FloatTensor(res))
 
    return res


def word_to_idx(data, embeddings):
    res = []
    for seq in range(len(data)):
        res.append(prepare_word_sequence(data[seq][0], embeddings))
    return res



def tag_to_idx(tags, tag2idx):
    res = []
    for sent in tags:
        temp_sent = []
        for tag in sent:
            if tag != '\n':
                tag = tag.rstrip()
                temp_sent.append([tag2idx[tag]])
        res.append(torch.LongTensor(temp_sent))
    return res



# extra data to be removed so that it can be divided in equal batches
def remove_extra(data, batch_size):
    extra = len(data) % batch_size
    if extra != 0:
        data = data[:-extra][:]
    return data


def combine_data(features, indexed_topics, indexed_words):
    res = []
    for i in range(len(features)):
        res.append((features[i], indexed_topics[i], indexed_words[i]))

    return res


def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_seqs, topics, sent = zip(*list_of_samples)

    input_seq_lengths = [len(seq) for seq in input_seqs]
    input_sent_lengths = [len(seq) for seq in sent]

    padding_value = 0

    # pad input sequences
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)
    
    # pad the sentences
    pad_sent_seqs = pad_sequence(sent, padding_value=padding_value)


    topics = torch.LongTensor(topics)
    
    return pad_input_seqs, input_seq_lengths, topics, pad_sent_seqs, input_sent_lengths
