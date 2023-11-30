from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import umap


def dim_red(mat,cor, p, method):




    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 

        p : number of dimensions to keep
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    if method=='ACP':
        red_mat = mat[:,:p]

        pca = PCA(n_components=20)
	red_mat = pca.fit_transform(mat)
    elif method=='TSNE':
      	red_mat = mat[:,:p]
	# Vectorize the text data using TF-IDF
	vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
	X = vectorizer.fit_transform(cor)

	# Apply t-SNE for dimensionality reduction
	tsne = TSNE(n_components=3, random_state=42)
	red_mat = tsne.fit_transform(X.toarray())

    elif method=='UMAP':
        red_mat = mat[:,:p]
        umap_model =umap.UMAP()  # Use umap.UMAP to create an instance of the UMAP class
    	red_mat= umap_model.fit_transform(mat)
    else:
        raise Exception("Please select one of the three methods : APC, AFC, UMAP")
    
    return red_mat


def clust(mat, k):
    '''
    Perform clustering

    Input:
    -----
        mat : input list 
        k : number of cluster
    Output:
    ------
        pred : list of predicted labels
    '''
    
    pred = np.random.randint(k, size=len(mat))
    
    return pred

# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# Perform dimensionality reduction and clustering for each method

methods = ['ACP', 'TSNE', 'UMAP']
for method in methods:
    # Perform dimensionality reduction
    red_emb = dim_red(embeddings,corpus, 20, method)

    # Perform clustering
    pred = clust(red_emb, k)

    # Evaluate clustering results
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)

    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')

