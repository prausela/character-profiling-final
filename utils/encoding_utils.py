import numpy as np
import pandas as pd

from gensim.models import Word2Vec as w2v
from sklearn.decomposition import PCA

def index(collection):
    from_index  = dict()
    to_index    = dict()

    i = 0
    for item in collection:
        from_index[i] = item
        to_index[item] = i
        i += 1

    return (from_index, to_index)

def one_hot_token(token, to_index):
    hot_encoded_token = np.zeros(shape=(1, len(to_index)))
    hot_encoded_token[0][to_index[token]] = 1
    return hot_encoded_token

def one_hot_sentence(tokenized_sentence, to_index):
    hot_encoded_sentence = np.zeros(shape=(1, len(to_index)))
    for token in tokenized_sentence:
        hot_encoded_sentence[0][to_index[token]] = 1
    return hot_encoded_sentence

def one_hot_document(tokenized_sentences, to_index):
    hot_encoded_sentences = np.zeros(shape=(len(tokenized_sentences), len(to_index)))
    
    i = 0
    for tokenize_sentence in tokenized_sentences:
        for token in tokenize_sentence:
            hot_encoded_sentences[i][to_index[token]] = 1
        i += 1

    return hot_encoded_sentences

def list2ndarray(hot_encoded_stuff):
    arr = np.zeros(shape=(len(hot_encoded_stuff), len(hot_encoded_stuff[0][0])))

    for i in range(0, len(hot_encoded_stuff)):
        arr[i] = hot_encoded_stuff[i][0]
    return arr

def w2v_embeddings_PCA(tokenized_doc, min_count=1, sg=1, window=7, pca_components=2, pca_random_state=7):
    
    w = w2v(
        tokenized_doc,
        min_count=min_count,  
        sg=sg,       
        window=window      
    )

    emb_df = (
        pd.DataFrame(
            [w.wv.get_vector(str(n)) for n in w.wv.key_to_index],
            index = w.wv.key_to_index
        )
    )

    pca = PCA(n_components=pca_components, random_state=pca_random_state)
    pca_mdl = pca.fit_transform(emb_df)

    emb_df_PCA = (
        pd.DataFrame(
            pca_mdl,
            columns=['x','y'],
            index = emb_df.index
        )
    )
    return emb_df_PCA
