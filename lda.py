import gensim
from utils import load_data
from gensim import corpora
import spacy
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import os
import pdb
from gensim.models import TfidfModel

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
    """
    Generate words for the lematized text
    Args:
        texts: texts
    return:
        final: a 2d list of lemmantization words
    """
    final = []
    #pdb.set_trace()
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return final






def idtoword(lem_words):
    """
    Thif function converts the text in 2d list to corpus and id2word dictionary for the LDA model as input
    Args:
        lem_words: A 2d list of words
    Return:
        corpus: corpus of words
        id2word: corpora of words for model
    """
    #pdb.set_trace()
    id2word = corpora.Dictionary(lem_words)
    
    corpus = []
    
    for text in lem_words:
        new = id2word.doc2bow(text)
        corpus.append(new)
        
        
    return id2word, corpus





def preprocess_data(texts):
    
    lem_texts = lemantization(texts)
    lem_words = gen_words(lem_texts)
    pdb.set_trace()
    id2word, corpus = idtoword(lem_words)
    
    return id2word, corpus
    
    
    
    
def LDA(id2word, corpus):

    
    model = gensim.models.ldamodel.LdaModel(
        corpus = corpus,
        id2word=id2word,
        num_topics=5,
        random_state=100,
        update_every =1,
        chunksize=100,
        passes=10,
        alpha='auto')
    
    vis = gensimvis.prepare(
        model, 
        corpus, 
        id2word, 
        mds="mds",
        R=30)
    
    pyLDAvis.save_html(vis, os.getcwd()+'/Results/ldavisbigramstrigrams.html')
    print("visualization saved in Results")
    return model
    


def make_bigrams_trigrams(data):
    """
    converts the list of words n*n into data bigrams and trigrams
    Args:
        data: 2d list of words
        data: 2d list of words: 2d list of words having data bigrams and trigrams
    """
    
    lem_text = lemantization(data)
    lem_words = gen_words(lem_text)
    
    bigram_phrases = gensim.models.Phrases(lem_words, min_count=5, threshold=50)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[lem_words], threshold=50)

    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)
    data_bigrams_trigrams = []
    
    for doc in lem_words:

        doc_bigrams = bigram[doc]
        doc_bigrams_trigrams = trigram[doc]
        data_bigrams_trigrams.append(doc_bigrams)
        
    return data_bigrams_trigrams
    

def tfidf_removal(data_bigrams_trigrams):
    """
    Removal of frequently occuring words 
    Args:
        data_bigrams_trigrams: 2d list of words of bigrams and trigrams
    Dependency:
        idtoword: which converts the data to corpus and corpora.Dictionary
    return:
        id2word: dictionary
        corpus: corpus of words Ready for the LDA Model as input
    
    """
    
    id2word, corpus = idtoword(data_bigrams_trigrams)
    
    tfidf = TfidfModel(corpus, id2word)
    low_value = 0.03
    words = []
    words_missing_in_tfidf = []
    
    for i in range(0, len(corpus)):
        
        bow = corpus[i]
        low_value_words = []
        
        tfidf_ids = [id for id, value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        drops = low_value_words + words_missing_in_tfidf
        
        for item in drops:
            words.append(id2word[item])
        
        words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids]
        
        new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
        
        corpus[i] = new_bow
        
    
    return id2word, corpus


def Sort(sub_li):
    
    sub_li.sort(key = lambda x: x[1])
    sub_li.reverse()
    
    return sub_li

