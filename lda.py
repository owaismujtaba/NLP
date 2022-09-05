import gensim
from utils import load_data
from gensim import corpora
import spacy
import pyLDAvis
import os
import pdb


def lemantization(texts, allowed_postages=['NOUN', 'ADJ', 'VERD', 'ADV']):

    
    #pdb.set_trace()
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    
    texts_out = []
    
    for text in texts:
        
        doc = nlp(text)
        new_text = []
        
        for token in doc:
            
            if token.pos_ in allowed_postages:
                
                if len(token.lemma_) > 3:
                    #print(token, token.lemma_)
                    new_text.append(token.lemma_)
            
        
        final = " ".join(new_text)
        
        texts_out.append(final)
        #
        
    return texts_out



def gen_words(texts):
    
    final = []
    #pdb.set_trace()
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return final



def idtoword(lem_words):
    id2word = corpora.Dictionary(lem_words)
    
    corpus = []
    
    for text in lem_words:
        new = id2word.doc2bow(text)
        corpus.append(new)
        
        
    return id2word, corpus


def preprocess_data(texts):
    
    lem_texts = lemantization(texts)
    lem_words = gen_words(lem_texts)
    id2word, corpus = idtoword(lem_words)
    
    return id2word, corpus
    
def LDA(id2word, corpus):
    
    #pdb.set_trace()
    model = gensim.models.ldamodel.LdaModel(
        corpus = corpus,
        id2word=id2word,
        num_topics=5,
        random_state=100,
        update_every =1,
        chunksize=100,
        passes=10,
        alpha='auto')
    
    vis = pyLDAvis.gensim.prepare(
        model, corpus, id2word, mds="mds", R=30)
    pyLDAvis.save_html(vis, os.getcwd()+/'Results/ldavis.html')
    print("visualization saved in Results")
    
    
    
                                        