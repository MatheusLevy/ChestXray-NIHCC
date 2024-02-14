import pandas as pd
from configs import *
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2
import pickle

def pre_process_csv(csv_path):
    df = pd.read_csv(csv_path, usecols=FIELDS)
    df['Image Index'] = df['Image Index'].apply(lambda img_filename: os.path.join(IMG_PATHS, img_filename))
    return df

def pre_process_img(img_path):
    img = cv2.imread(img_path)
    resized_image = cv2.resize(img, (WIDTH, HEIGHT))[...,::-1]
    return resized_image

def split_labels(dict_list):
    labels = []
    for item in dict_list:
        if 'Finding Labels' in item:
            labels_str = item['Finding Labels']
            labels_list = labels_str.split('|')
            labels.append(labels_list)
    return labels

def exclude_labels(dict_list, labels=[]):
    filter = [item for item in dict_list if not any(label in item.get('Finding Labels', '') for label in labels)]
    return filter

def make_pacient_dict(df):
    dict_list= df.to_dict(orient='records')
    dict_list = exclude_labels(dict_list, EXCLUDE_LABELS)
    labels= split_labels(dict_list)
    assert len(dict_list) == len(labels)
    return dict_list, labels

def get_unique_labels(labels_list):
    lista_1d = [label for sublist1 in labels_list for sublist2 in sublist1 for label in sublist2]
    unique_values = set(lista_1d)
    n_classes = len(unique_values)
    names = list(unique_values)
    return n_classes, names

def read_dataset():
    df= pre_process_csv(CSV_PATH)
    patients_id= df['Patient ID'].unique()
    X, y = [], []
    for id in tqdm(patients_id):
        patient_df = df.loc[df['Patient ID'] == id]
        pacients, labels= make_pacient_dict(patient_df)
        X.append(pacients)
        y.append(labels)
    
    X= [sublista for sublista in X if sublista]
    y= [sublista for sublista in y if sublista]
    assert len(X) == len(y)
    return X, y

def split_data(X, y, test_size=0.2, validation_size=0.25, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    validation_size = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=random_state, shuffle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test

def flatten_dataset(X, y):
    X_flatten, y_flatten= [], []
    for list_of_info_pacient, list_of_labels_pacient in zip(X,y):
        for dict_pacient, label_pacient in zip(list_of_info_pacient, list_of_labels_pacient):
            path = dict_pacient['Image Index']
            X_flatten.append(path)
            y_flatten.append(label_pacient)
    assert len(X_flatten) == len(y_flatten)
    return X_flatten, y_flatten

def save_history(history, model_path, branch="geral"):
    with open(f"{model_path}/history_{branch}", "wb") as f:
            pickle.dump(history, f)