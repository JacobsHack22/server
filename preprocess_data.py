import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import typing as t
import pickle
import time 
from sklearn.utils import shuffle

def get_classes(src_path: str) -> None:
    classes = []
    for instance in os.listdir(src_path):
        path = os.path.join(src_path, instance)
        if os.path.isdir(path):
            classes.append(path)
            
    classes = sorted(classes, key=lambda x: int(x.split(f'{src_path}/leaf')[1]))

    return classes
    
def augment_data(classes, num_rotations=3):
    for class_ in classes:
        for img in tqdm(os.listdir(class_)):
            if img != '.DS_Store':
                img_array = cv2.imread(os.path.join(class_, img))
                for i in range(num_rotations):
                    rotated = np.rot90(img_array, (i + 1))
                    cv2.imwrite(os.path.join(class_, f'{img[:-4]}_{i}.jpg'), rotated)

                flipped = cv2.flip(img_array, 0)
                cv2.imwrite(os.path.join(class_, f'{img[:-4]}_flipped.jpg'), flipped)

def create_training_data(classes) -> t.List[t.Tuple[np.ndarray, int]]:
    train_data = []
    for index, class_label in enumerate(classes):
        for img in tqdm(os.listdir(class_label)):
            if img != '.DS_Store':
                img_array = cv2.imread(os.path.join(class_label, img), 0)
                new_array = cv2.resize(img_array, (224, 224))
                train_data.append([new_array, index])
            
    return train_data
            
def split_data():
    classes_ = get_classes('data')
    training_data = create_training_data(classes_)

    X_data = []
    y_data = []

    for features, label in training_data:
        X_data.append(features)
        y_data.append(label)

    X_data = np.array(X_data).reshape(-1,224,224,1)
    
    return X_data, y_data

def preprocess_images(X_data, y_data):
    for i in range(len(X_data)):
        blur = cv2.GaussianBlur(X_data[i], (5,5),0)
        _, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)        
        kernel = np.ones((3,3), np.uint8)
        closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
        X_data[i] = closing.reshape(224, 224, 1)

        cv2.imwrite(f'data_embs/{y_data[i]}/{i}.jpg', X_data[i])

def prepare_and_save_data():
    X_data, y_data = split_data()
    preprocess_images(X_data)

    X = np.array(X_data / 255.0)
    y = np.array(y_data)
    X, y = shuffle(X, y)

    pickle_out = open("x_data_prepared","wb")
    pickle.dump(X_data, pickle_out)
    pickle_out.close()

    pickle_out = open("y_data_prepared","wb")
    pickle.dump(y_data, pickle_out)
    pickle_out.close()

def save_preprocessed_for_embeddings():
    X_data, y_data = split_data()
    preprocess_images(X_data, y_data)

if __name__ == '__main__':
    save_preprocessed_for_embeddings()