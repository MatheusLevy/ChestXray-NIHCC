import pandas as pd
from configs import *
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import json

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

def evaluate_classification_model(y_true, predictions, labels):
    auc_scores = roc_auc_score(y_true, predictions, average=None)
    auc_score_macro = roc_auc_score(y_true, predictions, average='macro')
    auc_score_micro = roc_auc_score(y_true, predictions, average='micro')
    auc_score_weighted = roc_auc_score(y_true, predictions, average='weighted')
    results = {
    "groun_truth" : y_true,
    "predictions" : predictions,
    "labels" : labels,
    "auc_scores" : auc_scores,
    "auc_macro" : auc_score_macro,
    "auc_micro" : auc_score_micro,
    "auc_weighted" : auc_score_weighted,
    }
    return results

def add_data(path, uuid, result):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)
    
    with open(path, 'r+') as f:
        dados = json.load(f)
        dados[uuid] = result
        f.seek(0)
        json.dump(dados, f)
        f.truncate()

def store_test_metrics(var, path, filename="test_metrics", name="", json=False):
    with open(f"{path}/{filename}", "wb") as f:
     pickle.dump(var, f) #salva arquivo
    if(json==True):
        auc_macro = var['auc_macro']
        add_data(path="results.json", uuid=name, result=auc_macro)
        pass

def get_pacients_id(X):
    ids = []
    for pacient_info in X:
        for pacient_img_ann in pacient_info:
            ids.append(pacient_img_ann['Patient ID'])
    unique_id = set(ids)
    return list(unique_id)

def validate_dataloaders(train_dataloader, val_dataloader, test_dataloader):
    train_x= train_dataloader.X_debug
    val_x= val_dataloader.X_debug
    test_x = test_dataloader.X_debug
    unique_ids_train = get_pacients_id(train_x)
    unique_ids_val = get_pacients_id(val_x)
    unique_ids_test = get_pacients_id(test_x)
    print(f'Len unique ids Train: {len(unique_ids_train)}')
    print(f'Len unique ids Val: {len(unique_ids_val)}')
    print(f'Len unique ids Test: {len(unique_ids_test)}')
    has_intersection = set(unique_ids_train) & set(unique_ids_val) & set(unique_ids_test)
    print(f'Patiente Intersection has: {has_intersection}')