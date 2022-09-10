from utils import load_data, drop_meaningless
from top2vec import Top2Vec
from nltk.corpus import stopwords

import os
import pdb



DATA_PATH = os.getcwd()+"/Data/papers.csv" 



def new_style():
    
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    
    from lda import lemantization
    
    data =  load_data(DATA_PATH)
    data = drop_meaningless(data)
    corpus = data['paper_text'].tolist()
    pdb.set_trace()
    corpus_lemantized = lemantization(corpus)
    
    
    # converting text to count vectorizer which contains frequency of words 
    
    count_vectorizer = CountVectorizer(stop_words=stopwords.words("english"), lowercase=True)
    word_counts = count_vectorizer.fit_transform(corpus_lemantized)
    
    word_counts = word_counts.todense()
    tfidf_transformer = TfidfTransformer()
    word_counts_tfidf = tfidf_transformer.fit_transform(x_counts)

    
def toptovec():
    
    
    data =  load_data(DATA_PATH)
    data = drop_meaningless(data)
    data = data['paper_text'].tolist()
    
    model = Top2Vec(data)
    pdb.set_trace()
    
    