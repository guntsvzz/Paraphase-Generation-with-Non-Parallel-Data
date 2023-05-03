import benepar, spacy
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

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

#### Dictionary + Dependency
new_dep = ['ACL', 'ACOMP', 'ADVCL', 'ADVMOD', 'AGENT', 'AMOD', 'APPOS', 'ATTR', 'AUX', 'AUXPASS', 'CASE','CC', 'CCOMP', 'COMPOUND', 'CONJ', 'CSUBJ', 'CSUBJPASS', 'DATIVE','DEP','DET', 'DOBJ', 'EXPL', 'INTJ', 'MARK', 'META','NEG', 'NOUNMOD', 'NPMOD', 'NSUBJ', 'NSUBJPASS', 'NUMMOD', 'OPRD', 'PARATAXIS', 'PCOMP', 'POBJ', 'POSS', 'PRECONJ', 'PREDET', 'PREP', 'PRT', 'PUNCT', 'QUANTMOD', 'RELCL', 'ROOT', 'XCOMP']
deprecated = ['COMPLM', 'INFMOD', 'PARTMOD', 'HMOD', 'HYPH', 'IOBJ', 'NUM', 'NUMBER', 'NMOD','NN', 'NPADVMOD', 'POSSESSIVE', 'RCMOD']
dependency_tags = new_dep + deprecated

import pickle

with open("./data/dictionary.pkl", "rb") as file:
    dictionary = pickle.load(file)
for i in range(31414, 31414+len(dependency_tags)):
    # print(i,dependency_tags[i-31414])
    dictionary.idx2word[i] = dependency_tags[i-31414].upper()
    a = dependency_tags[i-31414].upper()
    dictionary.word2idx[a] = i 

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
    return sent_, synt_

# ss = syntsent('I love sushi')
# print(ss)