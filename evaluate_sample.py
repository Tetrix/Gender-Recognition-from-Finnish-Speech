import torch
import torch.nn.functional as F

import numpy as np
import sys
import argparse

import utils.prepare_data as prepare_data
from config.config import *
from model import Transformer


# get the command line arguments
parser = argparse.ArgumentParser(description='Gender classification.')
parser.add_argument("--input", help="path to the input feature.")

args = parser.parse_args()

features = np.load(str(args.input))
#print(args.input)


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the features and transform them to the desired format
#features = np.load('lahjoita-puhetta-1-100.npy')
features = torch.autograd.Variable(torch.FloatTensor(features)).to(device)
features = features[:max_len]

# load the model
transformer = Transformer(features.size(1), 2, n_head_encoder, d_model, n_layers_encoder, max_len).to(device)
checkpoint = torch.load('weights/gender/state_dict_34.pt', map_location=torch.device('cpu'))
transformer.load_state_dict(checkpoint['transformer'])


features = features.unsqueeze(1)

# pass the data through the model
output = transformer(features, None, None, None)


# get the prediction
output = F.log_softmax(output, dim=-1)
topi, topk = output.topk(1)

if topk.item() == 0:
    gender = 'Mies'
else:
    gender = 'Nainen'

print(gender)


