import pandas as pd
import os
import re
import pdb
import nltk
import gensim
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





def test_model():
    
    # load the saved model
    model_path = os.getcwd()+"/Models/ldamodel.model"
    model = gensim.models.ldamodel.LdaModel.load(model_path)
    print("Model Loaded Scucessfully")
    
    return model
    
    
    
    
    
    
def clean_text(documents):
    """
    The function cleans the text in the documents:
    Args:
        documents: A list of documnets
    """
    from gensim.utils import simple_preprocess
    from spacy import load
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('punkt')
    
    

    stops = stopwords.words('english')
    
    
    for doc in documents:
        
        words = word_tokenize(doc)
        
        words = [w.lower() for w in words if not w.lower() in stops]
        
        pdb.set_trace()
    
    
    
    '''
    nlp = load("en_core_web_sm")
   
    cleaned_documents = []
    for doc in documents:
        cleaned_doc = []
        words_in_doc = nlp(doc)        
        for word in words_in_doc:
            if word.tag_ in ['NNP', 'VBZ', 'VBG', 'IN', 'NN']:
                cleaned_doc.append(word)
        
        cleaned_doc = [w for w in cleaned_doc if not w.lower() in stops]
        
        cleaned_documents.append(cleaned_doc)
        pdb.set_trace()
    print(tags)
    print(len(tags))
    '''
    
        
        