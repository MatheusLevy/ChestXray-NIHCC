import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Layer, Input, Concatenate

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

def InceptionV3_Block(x, filters):
    conv1x1_1 = keras.layers.Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    conv1x1_2 = keras.layers.Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    conv3x3 = keras.layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')(conv1x1_2)

    conv1x1_3 = keras.layers.Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    conv5x5 = keras.layers.Conv2D(filters[4], (5, 5), padding='same', activation='relu')(conv1x1_3)

    maxpool = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    conv1x1_4 = keras.layers.Conv2D(filters[5], (1, 1), padding='same', activation='relu')(maxpool)

    inception_block = keras.layers.concatenate([conv1x1_1, conv3x3, conv5x5, conv1x1_4], axis=-1)

    return inception_block


def classification_model_v1(input_shape, n_classes):
    backbone = keras.applications.DenseNet121(input_shape= input_shape, include_top= False, weights="imagenet") 
    backbone.trainable = True
    backbone_output = backbone.output

    x = keras.layers.Flatten()(backbone_output)
    layer = keras.layers.BatchNormalization()(x)
    layer = keras.layers.Dense(units=256, activation='relu', kernel_regularizer=l2(0.01))(layer)
    layer = keras.layers.Dropout(0.4)(layer)

    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Dense(units=128, activation='relu',kernel_regularizer=l2(0.01))(layer)
    layer = keras.layers.Dropout(0.4)(layer)
    
    output = keras.layers.Dense(units=n_classes, activation='sigmoid')(layer)
    model = keras.models.Model(inputs=backbone.input, outputs=output)
    return model

def classification_model_v2(input_shape, n_classes):
    img_input = Input(shape=input_shape)

    backbone_1 = keras.applications.ResNet50V2(input_shape= input_shape, include_top= False, weights="imagenet", input_tensor= img_input) 
    backbone_1.trainable = True
    backbone_1_output = backbone_1.output

    backbone_2 = keras.applications.ResNet50V2(input_shape= input_shape, include_top= False, weights="imagenet", input_tensor= img_input) 
    backbone_2.trainable = True
    for layer in backbone_2.layers:
        layer._name = layer._name + str("_2")
    backbone_2_output = backbone_1.output

    ramo_superior = InceptionV3_Block(backbone_2_output, [64, 128, 128, 64, 128, 64])
    ramo_inferior = InceptionV3_Block(backbone_1_output, [64, 128, 128, 64, 128, 64])

    concat = Concatenate()([ramo_inferior, ramo_superior])
    se_block_out = squeeze_excite_block(concat)

    x = keras.layers.Flatten()(se_block_out)
    layer = keras.layers.BatchNormalization()(x)
    layer = keras.layers.Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(layer)
    layer = keras.layers.Dropout(0.4)(layer)

    output = keras.layers.Dense(units=n_classes, activation='sigmoid')(layer)
    model = keras.models.Model(inputs=img_input, outputs=output)
    return model
