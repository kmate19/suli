import gensim
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk

# preprocess
nltk.download('punkt')
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  
    tokens = [t for t in tokens if t.isalpha()]  
    return tokens

# corpus
def prepare_corpus(data):
    return [preprocess_text(bio) for bio in data['bio']]

# train
def train_word2vec(corpus):
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")  
    return model

# vectorize
def text_to_embedding(text, model):
    tokens = preprocess_text(text)
    embeddings = [model.wv[word] for word in tokens if word in model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)  
    else:
        return np.zeros(model.vector_size)  

# similarity
def compute_similarity(data, model):
    data['bio_embedding'] = data['bio'].apply(lambda bio: text_to_embedding(bio, model))
    
    bio_embeddings = np.vstack(data['bio_embedding'].values)
    
    similarity_matrix = cosine_similarity(bio_embeddings)
    return similarity_matrix

# top matches
def find_top_matches(data, similarity_matrix, top_n=5):
    user_matches = {}
    for i, user in enumerate(data['username']):
        scores = similarity_matrix[i]
        sorted_indices = np.argsort(-scores)  
        
        top_indices = [idx for idx in sorted_indices if idx != i][:top_n]
        top_users = [(data['username'][idx], scores[idx]) for idx in top_indices]
        
        user_matches[user] = top_users
    return user_matches

# load data
file_path = 'tinder_data.csv'  
data = pd.read_csv(file_path)

corpus = prepare_corpus(data)

word2vec_model = train_word2vec(corpus)

similarity_matrix = compute_similarity(data, word2vec_model)

top_matches = find_top_matches(data, similarity_matrix, top_n=5)

# display
for user, matches in top_matches.items():
    print(f"\nTop matches for {user}:")
    for match, score in matches:
        print(f"  {match}: {score:.4f}")
