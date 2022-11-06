import pickle
from sentence_transformers import SentenceTransformer, util
import glob
from PIL import Image
import math
import cv2
import numpy as np
import os

def load_data():
    pickle_in = open("x_data_prepared","rb")
    X = pickle.load(pickle_in)

    pickle_in = open("y_data_prepared","rb")
    y = pickle.load(pickle_in)

    return X, y

def save_embeddings():
    img_model = SentenceTransformer('clip-ViT-B-32')
    files1 = glob.glob('data_embs/0/*.jpg')
    files2 = glob.glob('data_embs/1/*.jpg')
    files3 = glob.glob('data_embs/2/*.jpg')
    files = files1 + files2 + files3

    images = [Image.open(file) for file in files]
    embeddings = img_model.encode(images, convert_to_tensor = True)

    np.save('embeddings.npy', embeddings)

img_model = SentenceTransformer('clip-ViT-B-32')
emb = np.load('embeddings.npy')

# query = img_model.encode(Image.open('govno_lo.jpg'), convert_to_tensor=True)
# lst = util.semantic_search(query_embeddings=query, corpus_embeddings=emb, top_k=10)
# dicts = lst[0]
# for dic in dicts:
#     print(str(dic['corpus_id']) + '   ' + str(dic['score']))
