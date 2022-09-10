import os
import pandas as pd
import spacy
from nltk.corpus import stopwords
import string
from gensim import corpora
import gensim
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pdb


def clean_documents(documents):
    """
    Clean the raw text in the documents and retain only the necessary tokens
    Args:
        documents: A 2d list of textual documents
        
    Return:
        cleaned_docs: A 2d list of cleaned documents
    """
    print("Cleaning documents ")
    nlp = spacy.load('en_core_web_sm')
    stops = stopwords.words('english')
    alphabet = list(string.ascii_lowercase)
    stops = stops + alphabet
    
    cleaned_docs = []
    
    for doc in documents:
        doc = nlp(doc)
        tokens =[]
        
        for token in doc:
            # Keeping only nouns verbs adjectives and whose length is greater than 2
            if token.tag_ in ['NNP', 'VBZ', 'VBG', 'IN', 'ADJ'] and str(token.text).lower() not in stops and len(token.text)>2:
                tokens.append(token.lemma_.lower())
        
        cleaned_docs.append(tokens)
    
    
    print("Documents cleaned")   
    return cleaned_docs



def idtoword(id2word, documents):
    
    """
    Gives the corpus from the documents
    Args:
        id2word: corpora Dictionary fitted on documents
        documents: A 2d list of cleaned documents containing tokens in each document
        
    Return:
        corpus: Corpus of words
    """
    
    corpus = []
    
    for text in documents:
        new = id2word.doc2bow(text)
        corpus.append(new)
    return corpus
    
    
    
def LDA(documents, n_topics=5):
    
    """
    Fit and LDA model on the data
    Args:
        documents: A 2d list of cleaned documents containing tokens in each document
        n_topics: number of topics or clusters
    
    """
    print("Fitting the LDA Model")
    id2word = corpora.Dictionary(documents)
    corpus = idtoword(id2word, documents)
    
    model = gensim.models.ldamodel.LdaModel(
        corpus = corpus,
        id2word=id2word,
        num_topics=n_topics,
        random_state=100,
        update_every =1,
        chunksize=100,
        passes=10,
        alpha='auto')
    
    vis = gensimvis.prepare(
            model, 
            corpus, 
            id2word, 
            mds="PCoA",
            R=30)

    pyLDAvis.save_html(vis, os.getcwd()+'/Results/lda.html')
    print("Results saved in Results Folder")
    
    model_path = os.getcwd()+'/Models/LDAmodel.model'
    
    model.save(model_path)
    
    print("Model Saved: ", model_path)
    cluster_test(corpus, model)


def pipeline_LDA(raw_documents, n_topics=5):
    
    """
    Applies LDA Algorithm on the raw documents and clusters them in by default 5 topics
    Args:
        raw_documents: A 2d list of raw textual documents
        n_topics: Number of topics
    """
    documents = clean_documents(raw_documents)
    documents = make_bigrams_trigrams(documents)
    LDA(documents, n_topics)

    
    
    
def get_similarity(documents):
    
    documents = clean_documents(documents)
    documents = make_bigrams_trigrams(documents)
    
    model = load_model()
    
    
    
    
    
def make_bigrams_trigrams(documents):
    
    #pdb.set_trace()
    bigram_phrases = gensim.models.Phrases(documents, min_count=5, threshold=50)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[documents], min_count=5, threshold=50)
    
    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)
    
    bigram_trigram_documents = []
    for doc in documents:
        
        doc_bigrams = bigram[doc]
        doc_bigrams_trigrams = trigram[doc]
        bigram_trigram_documents.append(doc_bigrams_trigrams)
        
    
    return documents
        
    
    
def cluster_test(corpus, model):
    docs_with_1_topic = 0
    docs_with_multiple_topics = 0
    docs_with_no_topics = 0
    total_docs = 0
    for doc in corpus:
        topics = model.get_document_topics(doc, minimum_probability=0.20)
        total_docs += 1
        if len(topics) == 1:
            docs_with_1_topic += 1
        elif len(topics) > 1:
            docs_with_multiple_topics += 1
        else:
            docs_with_no_topics += 1
    print('Corpus assigned to a single topic:', (docs_with_1_topic / total_docs) * 100, '%')
    print('Corpus assigned to multiple topics:', (docs_with_multiple_topics / total_docs) * 100, '%')
    print('corpus assigned to no topics:', (docs_with_no_topics / total_docs) * 100, '%')
    

    
    
def load_model():
    
    model_path = os.getcwd()+'/Models/LDAmodel.model'
    model = gensim.models.LdaModel.load(model_path)
    
    return model
    

    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

