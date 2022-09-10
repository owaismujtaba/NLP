from utils import cleaned_data
from models import KMEANS
from tfidf import vectorize_data
from lda import preprocess_data, LDA, make_bigrams_trigrams, tfidf_removal
from utils import load_data, test_model
from t2v import toptovec
import os
import pdb
import gensim
import pandas as pd

import warnings


DATA_PATH = os.getcwd()+'/Data/papers.csv'
n_clusters=10
init='k-means++'
max_iter=100



def make_bigrams(lem_words):
    return(bigram[doc] for doc in lem_words) 

def make_trigrams(lem_words):
    return(trigram[bigram[doc]] for doc in lem_words)


print('1: TF-IDF \n2: LDA \n3: Bigrams, Trigrams')
choice = 12

if choice == 1:
    
    data = cleaned_data(PATH=DATA_PATH)
    vectorizer, vectors = vectorize_data(data)
    KMEANS(vectorizer, vectors, n_clusters, init, max_iter)

if choice == 2:
    data = load_data(DATA_PATH)
    data = data['paper_text']
    
    #data = make_bi_tri_grams(data)
    id2word, corpus = preprocess_data(data)
    lda_model = LDA(id2word, corpus)
    
    
if choice == 3:
    data = load_data(DATA_PATH)
    data = data['paper_text']
    
    data_bigrams_trigrams = make_bigrams_trigrams(data)
    
    id2word,  corpus = tfidf_removal(data_bigrams_trigrams)
    
    lda_model = LDA(id2word, corpus)
    lda_model.save(os.getcwd()+"/Models/ldamodel.model")

    
if choice ==4:
    model = test_model()
    
if choice ==5:
    
    toptovec()
if choice ==6:
    from t2v import new_style
    new_style()

    
if choice ==10:
    from utils import clean_text
    
    data = load_data(DATA_PATH)
    documents = data['paper_text']
    
    clean_text(documents)
    
    
if choice == 11:
    from final import clean_documents
    
    file = os.getcwd()+'/Data/papers.csv'
    data = pd.read_csv(file, nrows=10)

    data = data['paper_text']
    
    clean_documents(data)

    
if choice ==12:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        from final import pipeline_LDA
        from final import load_model
        
        load_model()
        file = os.getcwd()+'/Data/papers.csv'
        data = pd.read_csv(file, nrows=100)

        train = data['paper_text'][:90]

        pipeline_LDA(train, 10)