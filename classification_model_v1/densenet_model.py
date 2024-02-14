import tensorflow.keras as keras
import tensorflow as tf

def classification_model_v1(input_shape, n_classes):
    backbone = keras.applications.DenseNet121(input_shape= input_shape, include_top= False, weights="imagenet") 
    backbone.trainable = True
    backbone_output = backbone.output

    x = keras.layers.Flatten()(backbone_output)
    layer = keras.layers.BatchNormalization()(x)
    layer = keras.layers.Dense(units=256, activation='relu')(layer)
    layer = keras.layers.Dropout(0.3)(layer)

    layer = keras.layers.BatchNormalization()(x)
    layer = keras.layers.Dense(units=128, activation='relu')(layer)
    layer = keras.layers.Dropout(0.3)(layer)

    output = keras.layers.Dense(units=n_classes, activation='sigmoid')(layer)
    model = keras.models.Model(inputs=backbone.input, outputs=output)
    return model

def f1_score(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.equal(y_true, 1), tf.math.equal(y_pred, 1)), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(tf.math.equal(y_pred, 1), tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(tf.math.equal(y_true, 1), tf.float32))

    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1

def hamming_loss(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.not_equal(y_true, tf.round(y_pred)), tf.float32))
