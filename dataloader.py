
import tensorflow as tf
from utils import pre_process_img, np
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
from utils import flatten_dataset
from sklearn.utils import shuffle

class Chest_DataLoader(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, num_classes, labels_names):
        X_train_flatten, y_train_flatten = flatten_dataset(X, y)
        self.X_debug = X
        self.labels_name= labels_names
        self.paths = X_train_flatten
        self.labels = y_train_flatten
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.indexes = list(range(len(self.paths)))
        self.label_binarizer = MultiLabelBinarizer()
        self.label_binarizer.fit([labels_names])

    def __len__(self):
        return int(tf.math.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_data = self.paths[start:end]
        batch_labels = self.labels[start:end]
        # batch_debug = self.X[start:end]
        X, y = self.__load_data(batch_data, batch_labels)
        return X, y
    def get_labels(self):
        labels_binarized = []
        for label in self.labels:
            labels_binarized.append(self.label_binarizer.transform([label])[0])
        return np.asarray(labels_binarized)

    def on_epoch_end(self):
        tf.random.shuffle(self.indexes)

    def normalize_img(self, img):
        img = img.astype(np.float32)
        img_normalized = img/255.0
        return img_normalized
    
    def __load_data(self, batch_data, batch_labels):
        X, y= [], []
        
        for data_point, label in zip(batch_data, batch_labels):
            img = pre_process_img(data_point)
            img_array = self.normalize_img(img)
            X.append(img_array)
            one_hot_label = self.label_binarizer.transform([label])[0]
            y.append(one_hot_label)
        return tf.convert_to_tensor(X), tf.convert_to_tensor(y)