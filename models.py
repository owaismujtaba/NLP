from tfidf import vectorize_data
from sklearn.cluster import KMeans
from vis_utils import plot_kmeans
import os

import pdb

def KMEANS(vectorizer, vectors, n_clusters=10, init='k-means++', max_iter=100):
    
    
    
    
    
    model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter)
    
    model.fit(vectors)
    
    
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    
    
    
    with open (os.getcwd()+'/Results/kmeans.txt', 'w', encoding='utf-8') as f:
        
        for i in range(n_clusters):
            f.write(f"Cluster {i}")
            f.write("\n")

            
            for ind in order_centroids[i, :10]:
                f.write(' %s' % terms[ind],)
                f.write("\n")

    f.close()      
    print("Results saved in the text file in Results Folder")
    
    plot_kmeans(model, vectors, n_clusters)
    
    