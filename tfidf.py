from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_data(papers):
    
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=100,
        max_df=0.8,
        min_df=5,
        ngram_range=(1,3),
        stop_words="english"  
    )
    
    texts = papers['text_processed']
    
    vectors = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names()
    
    #print(feature_names)
    dense = vectors.todense()
    denselist = dense.tolist()
    
    all_keywords = []
    
    for description in denselist:
        x = 0
        keywords = []
        
        for word in description:
            if word>0:
                keywords.append(feature_names[x])
            x = x + 1
        
        all_keywords.append(keywords)
    
    
        
    return vectorizer, vectors 
    
    



