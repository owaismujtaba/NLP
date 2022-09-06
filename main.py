from utils import cleaned_data
from models import KMEANS
from tfidf import vectorize_data
from lda import preprocess_data, LDA, make_bigrams_trigrams, tfidf_removal
from utils import load_data

import os
import pdb
import gensim

DATA_PATH = os.getcwd()+'/Data/papers.csv'
n_clusters=10
init='k-means++'
max_iter=100


def make_bigrams(lem_words):
    return(bigram[doc] for doc in lem_words) 

def make_trigrams(lem_words):
    return(trigram[bigram[doc]] for doc in lem_words)


print('1: TF-IDF \n2: LDA \n3: Bigrams, Trigrams')
choice = 3

if choice == 1:
    
    data = cleaned_data(PATH=DATA_PATH)
    vectorizer, vectors = vectorize_data(data)
    KMEANS(vectorizer, vectors, n_clusters, init, max_iter)

if choice == 2:
    data = load_data(DATA_PATH)
    data = data['paper_text']
    
    data = make_bi_tri_grams(data)
    id2word, corpus = preprocess_data(data)
    LDA(id2word, corpus)
    
    
if choice == 3:
    data = load_data(DATA_PATH)
    data = data['paper_text']
    
    data_bigrams_trigrams = make_bigrams_trigrams(data)
    
    id2word,  corpus = tfidf_removal(data_bigrams_trigrams)
    
    LDA(id2word, corpus)
