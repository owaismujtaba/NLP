from utils import cleaned_data
from models import KMEANS
from tfidf import vectorize_data
from lda import preprocess_data, LDA
from utils import load_data
import os
import pdb


DATA_PATH = os.getcwd()+'/Data/papers.csv'
n_clusters=10
init='k-means++'
max_iter=100



print('1: TF-IDF')
choice = 2

if choice == 1:
    
    data = cleaned_data(PATH=DATA_PATH)
    vectorizer, vectors = vectorize_data(data)
    KMEANS(vectorizer, vectors, n_clusters, init, max_iter)

if choice == 2:
    data = load_data(DATA_PATH)
    data = data['paper_text']
    id2word, corpus = preprocess_data(data)
    LDA(id2word, corpus)
    
    pdb.set_trace()
    