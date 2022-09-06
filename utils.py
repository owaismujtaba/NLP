import pandas as pd
import os
import re
import pdb
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer



   
    
def load_data(PATH):
    
    papers = pd.read_csv(PATH, nrows=10)
    return papers

    
def drop_meaningless(papers):
    
    meaningless_features = ['id', 'event_type', 'pdf_name']
    papers.drop(meaningless_features, axis=1, inplace=True)
    
    return papers


def text_cleaning(papers):
    
    
    
    papers['text_processed'] = papers['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x))
    
    papers['text_processed'] = papers['paper_text'].map(lambda x: x.lower())
    
    stops = stopwords.words('english')
    stops = stops + ['ii', 'fig', 'et', 'al', 'figure']
    
    papers['text_processed'] = papers['text_processed'].map(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))
    #pdb.set_trace()
    
    papers['text_processed'] = papers['text_processed'].map(lambda x: ' '.join([word for word in x.split() if len(word) >1]))
    papers['text_processed'] = papers['text_processed'].map(lambda x: x.replace(string.punctuation, ""))
    papers['text_processed'] = papers['text_processed'].replace("  ", " ")
    
    #pdb.set_trace()
    
    papers['text_processed'] = papers['text_processed'].str.replace('\d+', '')
    return papers

    

def cleaned_data(PATH):
    
    data = load_data(PATH)
    data = drop_meaningless(data)
    data = text_cleaning(data)
    
    return data


