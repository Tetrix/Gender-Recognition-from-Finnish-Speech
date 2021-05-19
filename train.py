import random
import torch
import torch.nn.functional as F
import numpy as np
from utils.calculate_f1 import print_scores
from utils.beam_search import Generator
from jiwer import wer

import pickle
import gc

def train(pairs_batch_train, pairs_batch_dev, transformer, criterion, optimizer, num_epochs, batch_size, train_data_len, dev_data_len, device):
    accumulate_steps = 3
    
    for epoch in range(27, 51):
        all_predictions = []
        all_labels = []
        all_tag_predictions = []
        all_tags = []

        transformer.train()
        
        batch_loss_train = 0 
        batch_loss_dev = 0
    
        
        # gradient accumulation
        optimizer.zero_grad()
        for iteration, batch in enumerate(pairs_batch_train):
            train_loss = 0
            
            pad_input_seqs, input_seq_lengths, topic_seqs, pad_sent_seqs, input_sent_lengths = batch
            pad_input_seqs, topic_seqs, pad_sent_seqs = pad_input_seqs.to(device), topic_seqs.to(device), pad_sent_seqs.to(device)
            
            output = transformer(pad_input_seqs, input_seq_lengths, pad_sent_seqs, input_sent_lengths)

            train_loss = criterion(output, topic_seqs)
            train_loss = train_loss / accumulate_steps
            train_loss.backward()
            batch_loss_train += train_loss.detach().item()

            if (iteration + 1) % accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()


        
        # VALIDATION
        with torch.no_grad():
            transformer.eval()

            for iteration, batch in enumerate(pairs_batch_dev):
                dev_loss = 0

                pad_input_seqs, input_seq_lengths, topic_seqs, pad_sent_seqs, input_sent_lengths = batch
                pad_input_seqs, topic_seqs, pad_sent_seqs = pad_input_seqs.to(device), topic_seqs.to(device), pad_sent_seqs.to(device)
            
                output = transformer(pad_input_seqs, input_seq_lengths, pad_sent_seqs, input_sent_lengths)

                
                dev_loss = criterion(output, topic_seqs)
                batch_loss_dev += dev_loss.item()
               


        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch, (batch_loss_train / len(pairs_batch_train)), (batch_loss_dev / len(pairs_batch_dev))))
        
        #with open('loss/gender_new.txt', 'a') as f:
        #    f.write(str(epoch) + '  ' + str(batch_loss_train / len(pairs_batch_train)) + '  ' + str(batch_loss_dev / len(pairs_batch_dev)) + '\n')

        #
        #print('saving the model...')
        #torch.save({
        #'transformer': transformer.state_dict(),
        #'optimizer': optimizer.state_dict(),
        #}, 'weights/gender_new/state_dict_' + str(epoch) + '.pt')



