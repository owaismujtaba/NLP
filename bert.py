from bertopic import BERTopic
from spacy import load
from nltk.corpus import stopwords
import string
import pdb
import os
import pandas as pd

def clean_text_bert(documents):
    
    stops = stopwords.words('english')
    nlp = load('en_core_web_sm')
    alphabet = list(string.ascii_lowercase)
    stops = stops + alphabet
    cleaned_documents = []
    for doc in documents:
        doc = nlp(doc)
        cleaned_doc = ''
        for token in doc[2000:3500]:
            
            if token.tag_ not in ['SYM', 'NUM', 'CD', '_SP', 'NNS', '.', 'xx', 'ADD'] and str(token.text).lower() not in stops and len(token.text)>2:
                #print("token : ",token.lemma_)
                #print('Tag : ',  token.tag_)
                cleaned_doc = cleaned_doc + ' '  + str(token.lemma_).lower()
        cleaned_documents.append(cleaned_doc)
    #pdb.set_trace()
        
    
    return cleaned_documents
    


def bert_model(texts):
    
    model = BERTopic()
    model = BERTopic(embedding_model="all-MiniLM-L6-v2")
    topics, probs = model.fit_transform(texts)
    #pdb.set_trace()
    return model
    
'''    
file = os.getcwd()+'/Data/papers.csv'
data = pd.read_csv(file, nrows=100)
data = data['paper_text']

texts = clean_text_bert(data)
bert_model(texts)
'''