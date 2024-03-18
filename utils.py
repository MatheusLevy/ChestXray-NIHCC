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
from collections import defaultdict
from pulp import LpProblem, LpVariable, lpSum, value, LpBinary,LpMinimize
import math
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array

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

def read_dataset(path):
    df= pre_process_csv(path)
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

def store_test_metrics(var, path, filename="test_metrics", name="", json=False, result_path='result.json'):
    with open(f"{path}/{filename}", "wb") as f:
     pickle.dump(var, f) #salva arquivo
    if(json==True):
        auc_macro = var['auc_macro']
        add_data(path=result_path, uuid=name, result=auc_macro)
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

def calc_class_numbers(y, class_names, verbose=True):
    class_count = {_class: 0 for _class in class_names}
    for pacient in y:
        for img_labels in pacient:
            for label in img_labels:
                class_count[label] +=1
    if verbose:
        for _class, value in class_count.items():
            print(f'{_class}: {value}')
    return class_count
    
def calc_porcetage_pacient(pacient, class_count_dict, class_names):
    class_count_pacient = {_class: 0 for _class in class_names}
    for img_labels in pacient:
         for label in img_labels:
                class_count_pacient[label] +=1
    porcent = {classe: (class_count_pacient[classe] / class_count_dict[classe]) * 100 if class_count_dict[classe] != 0 else 0 for classe in class_count_pacient}
    return porcent

def compute_class_porcentage_dict(y_calc, y_total, class_names):
    porcentage_amostras = []
    dict_class_count = calc_class_numbers(y_total,class_names, verbose=False)
    for pacient in tqdm(y_calc):
        porcent= calc_porcetage_pacient(pacient, dict_class_count, class_names)
        porcentage_amostras.append(porcent)
    return porcentage_amostras

def divide_split(porcentage_A, porcentage_B, Total):
  value_A = (porcentage_A / 100) * Total
  value_B = (porcentage_B / 100) * Total
  value_A = int(value_A)
  value_B = int(value_B)

  if value_A + value_B != Total:
    value_B = Total - value_A
  return value_A, value_B
2
def get_generator_train(df, x_col, names, batch_size=16, shuffle=False, size=(256,256), imageDataGenerator=None, ):
    df['path'] = df.apply(lambda row: f'/home/matheus_levy/workspace/lucas/dataset/images/{row["Image Index"]}', axis=1) 
    datagen = imageDataGenerator
    if imageDataGenerator==None:
        datagen = ImageDataGenerator(
            horizontal_flip = True,
            rescale=1/255.0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rotation_range=10,
            shear_range=0.1,
            zoom_range=0.1,
            vertical_flip=False,
            fill_mode='nearest'
            )
        

    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory = None,
        x_col=x_col,
        y_col= names,
        class_mode= "raw",
        target_size=size,
        batch_size=batch_size,
        shuffle=shuffle,
        
    )
    return generator



def get_generator_val(df, x_col, names, batch_size=16, shuffle=False, size=(256,256), imageDataGenerator=None, ):
    df['path'] = df.apply(lambda row: f'/home/matheus_levy/workspace/lucas/dataset/images/{row["Image Index"]}', axis=1) 
    datagen = imageDataGenerator
    if imageDataGenerator==None:
        datagen = ImageDataGenerator(
            rescale=1/255.0
            )
        

    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory = None,
        x_col=x_col,
        y_col= names,
        class_mode= "raw",
        target_size=size,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return generator


def count_classes(df, names):
    class_counts = df[names].sum()
    no_class_count = len(df) - df[names].any(axis=1).sum()
    print("Número de amostras em cada classe:")
    print(class_counts)
    print("\nNúmero de amostras sem nenhuma das classes:", no_class_count)