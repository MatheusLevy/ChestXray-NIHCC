import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K

def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se =  keras.layers.GlobalAveragePooling2D()(init)
    se =  keras.layers.Reshape(se_shape)(se)
    se =  keras.layers.Dense(filters // ratio, activation='relu',
               kernel_initializer='he_normal', use_bias=False)(se)
    se =  keras.layers.Dense(filters, activation='sigmoid',
               kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se =  keras.layers.Permute((3, 1, 2))(se)

    x =  keras.layers.multiply([init, se])
    return x

def classification_model_v1(input_shape, n_classes):
    backbone = keras.applications.DenseNet121(input_shape= input_shape, include_top= False, weights="imagenet") 
    backbone.trainable = True
    backbone_output = backbone.output

    x = keras.layers.Flatten()(backbone_output)
    layer = keras.layers.BatchNormalization()(x)
    layer = keras.layers.Dense(units=256, activation='relu')(layer)
    layer = keras.layers.Dropout(0.4)(layer)

    layer = keras.layers.BatchNormalization()(x)
    layer = keras.layers.Dense(units=128, activation='relu')(layer)
    layer = keras.layers.Dropout(0.4)(layer)

    output = keras.layers.Dense(units=n_classes, activation='sigmoid')(layer)
    model = keras.models.Model(inputs=backbone.input, outputs=output)
    return model

def classification_model_v2(input_shape, n_classes):
    backbone = keras.applications.DenseNet121(input_shape= input_shape, include_top= False, weights="imagenet") 
    backbone.trainable = True
    backbone_output = backbone.output

    se_block_out = squeeze_excite_block(backbone_output)

    x = keras.layers.Flatten()(se_block_out)
    layer = keras.layers.BatchNormalization()(x)
    layer = keras.layers.Dense(units=256, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01), activity_regularizer=keras.regularizers.L2(0.01))(layer)
    layer = keras.layers.Dropout(0.2)(layer)

    layer = keras.layers.BatchNormalization()(x)
    layer = keras.layers.Dense(units=128, activation='relu', kernel_regularizer=keras.regularizers.L1(0.01), activity_regularizer=keras.regularizers.L2(0.01))(layer)
    layer = keras.layers.Dropout(0.2)(layer)

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
