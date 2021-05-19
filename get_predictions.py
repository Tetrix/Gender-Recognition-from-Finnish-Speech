import torch
from jiwer import wer
import numpy as np
import fastwer

import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def get_predictions(transformer, batch_size, idx2topic, test_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')
    
    transformer.eval()
    
    for i in range(1, 2):
        true_topics = []
        predicted_topics = []

        for l, batch in enumerate(test_data):
            #checkpoint = torch.load('weights/gender_new/state_dict_' + str(i) + '.pt', map_location=torch.device('cpu'))
            #transformer.load_state_dict(checkpoint['transformer'])
            #transformer.eval()


            pad_input_seqs, input_seq_lengths, topic_seqs, pad_sent_seqs, input_sent_lengths = batch
            pad_input_seqs, topic_seqs, pad_sent_seqs = pad_input_seqs.to(device), topic_seqs.to(device), pad_sent_seqs.to(device)
                
            predicted_indices = []

            #src_mask = transformer.generate_square_subsequent_mask(pad_input_seqs.size(0)).to(device)
            #src_padding_mask = (pad_input_seqs == 0).transpose(0, 1)[:, :, 0].to(device)
            
            print(pad_input_seqs.size())

            output = transformer(pad_input_seqs, input_seq_lengths, pad_sent_seqs, input_sent_lengths)
            output = F.log_softmax(output, dim=-1)
            topi, topk = output.topk(1)
            
            true_topics.append(topic_seqs.item())
            predicted_topics.append(topk.item())

            #if topic_seqs.item() != topk.item():
            #    wrong_trn = []
        
        score = precision_recall_fscore_support(true_topics, predicted_topics, average='micro')
        precision, recall, f1 = score[0], score[1], score[2]
        
        #print(predicted_topics)

        #true_topics = np.array(true_topics)
        #predicted_topics = np.array(predicted_topics)
        #a = np.where(true_topics != predicted_topics)
        #np.save('wrong_pred.npy', a)
        
        print('Epoch: %.d   Accuracy: %.4f' % (i, accuracy_score(true_topics, predicted_topics))) 
        #print('Precision: %.4f      Recall: %.4f        F1: %.4f' % (precision, recall, f1))

       
       
      
