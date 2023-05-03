import benepar, spacy
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
import torch
#### Configuration
def is_paren(tok):
    return tok == ")" or tok == "("

def getleaf(tree):
    nonleaves = ''
    for w in str(tree).replace('\n', '').split():
        w = w.replace('(', '( ').replace(')', ' )')
        nonleaves += w + ' '
    
    leaves = []
    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not is_paren(tok1) and not is_paren(tok2):
                leaves.append(arr[n])

    return leaves

def deleaf(tree):
    nonleaves = ''
    for w in str(tree).replace('\n', '').split():
        w = w.replace('(', '( ').replace(')', ' )')
        nonleaves += w + ' '

    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not is_paren(tok1) and not is_paren(tok2):
                arr[n + 1] = ""

    nonleaves = " ".join(arr)
    return nonleaves.split()

import pickle

with open("./data/dictionary.pkl", "rb") as file:
    dictionary = pickle.load(file)
vocab_dict = dictionary.word2idx

#### SyntSent
from nltk import ParentedTree
def syntsent(sentence):
    doc   = nlp(sentence.lower())
    ###Sentence Token
    sent  = list(doc.sents)[0]
    sent_ = [dictionary.word2idx[f'{w}'] for w in sent]
    ###Syntax Token
    synt  = ParentedTree.fromstring(sent._.parse_string)
    synt  = deleaf(synt)
    synt  = [dictionary.word2idx[f"<{w}>"] for w in synt if f"<{w}>" in dictionary.word2idx]
    synt_ = [dictionary.word2idx["<sos>"]] + synt + [dictionary.word2idx["<eos>"]]
    trg_ = [dictionary.word2idx[f'{w}'] for w in sent]
    trg_ = [dictionary.word2idx["<sos>"]] + trg_ + [dictionary.word2idx["<eos>"]]
    return torch.tensor(sent_).reshape(1,-1), torch.tensor(synt_).reshape(1,-1), torch.tensor(trg_).reshape(1,-1)

# ss = syntsent('I love sushi')
# print(ss[0].shape)
# print(ss[2].shape)
# print(ss[1].shape)