import pickle
from gensim.corpora import MmCorpus
from gensim.corpora import Dictionary
from collections import defaultdict

# Load date mappings
with open('data/date_mappings.pkl', 'rb') as f:
    date_mappings = pickle.load(f)

# Load dictionary and corpus
dictionary = Dictionary.load('tbmm_corpus.mm.tbmm_lda.model.id2word')
corpus = MmCorpus('tbmm_corpus.mm')

