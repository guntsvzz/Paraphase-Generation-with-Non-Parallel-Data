from model import model_adv, model_normal, model_synPG
import torch

from utils import load_embedding, reverse_bpe, sent2str, synt2str
from utils_sentence import *

input_dim   = len(vocab_dict)
emb_dim     = 300  #fasttext
word_dropout = 0.4 #following SynPG
dropout      = 0.1

glove_file = './data/glove.6B.300d.txt'
embedding = load_embedding(glove_file, dictionary)

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pad_idx = dictionary.word2idx['<pad>'] ##get the pad index from the vocab

def select_model(model):
    if model == 'normal':
        model = model_normal.Normal_Transformer(input_dim=input_dim, emb_dim = emb_dim, device=device, word_dropout = word_dropout, dropout = dropout)
        model = model.cuda()
        model.load_embedding(embedding)
        save_path = './model/nmt_paraphase_1m.pt'
        model.load_state_dict(torch.load(save_path))
    elif model == 'adv':
        model = model_adv.Adversarial_Transformer(input_dim=input_dim, emb_dim = emb_dim, device=device, word_dropout = word_dropout, dropout = dropout)
        model = model.cuda()
        model.load_embedding(embedding)
        save_path = './model/adversarial_nmt_paraphase_1m.pt'
        model.load_state_dict(torch.load(save_path))
    elif model == 'full':
        model = model_synPG.SynPG(vocab_size=input_dim, em_size = emb_dim, word_dropout = word_dropout, dropout = dropout)
        model = model.cuda()
        model.load_embedding(embedding)
        save_path = './model/pretrained_synpg.pt'
        model.load_state_dict(torch.load(save_path))
    return model

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

pad_idx = dictionary.word2idx['<pad>'] ##get the pad index from the vocab

from tqdm import tqdm
def generate(model, data, vocab_transform):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        batch_size   = data[0].size(0)
        max_sent_len = data[0].size(1)
        max_synt_len = data[1].size(1) - 2  # count without <sos> and <eos>

        # Put input into device
        sents_ = data[0].to(device)
        synts_ = data[1].to(device)
        trgs_ = data[2].to(device)

        # generate
        idxs = model.generate(sents_, synts_, sents_.size(1), temp=0.5) 
        
        for sent, idx, synt in zip(sents_.cpu().numpy(), idxs.cpu().numpy(), synts_.cpu().numpy()):
            convert_sent = reverse_bpe(sent2str(sent, vocab_transform).split()) + '\n'
            convert_idx = synt2str(idx, vocab_transform) +'\n'
            
            return convert_sent, convert_idx

# data = syntsent('I love sushi')
# model = select_model('full')
# target, output = generate(model, data, dictionary)

# print(target,output)