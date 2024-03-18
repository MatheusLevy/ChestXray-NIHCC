import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Layer, Input, Concatenate
from tensorflow.keras.applications import DenseNet121, DenseNet169, Xception
from tensorflow.keras.models import Sequential

def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = keras.layers.GlobalAveragePooling2D()(init)
    se = keras.layers.Reshape(se_shape)(se)
    se = keras.layers.Dense(filters // ratio, activation='relu',
               kernel_initializer='he_normal', use_bias=False)(se)
    se = keras.layers.Dense(filters, activation='sigmoid',
               kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = keras.layers.Permute((3, 1, 2))(se)

    x = keras.layers.multiply([init, se])
    return x

def densenet_pneumo(n_classes):
    return Sequential([
            DenseNet121(input_shape=(256, 256, 3), include_top=False, weights='imagenet'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(n_classes, activation='sigmoid')
    ])


def fusion_model(input_shape, n_classes):
    img_input = Input(shape=input_shape)

    densenet = tf.keras.applications.DenseNet121(
        include_top=False, weights='imagenet', input_tensor=img_input)
    
    for layer in densenet.layers:
        layer._name = layer._name + str("_2")
    densenet_out = densenet.output

    resnet = tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_tensor=img_input)
    resnet_out = resnet.output

    resent_branch_out = keras.layers.Conv2D(1024, 1, strides=1, padding='same')(resnet_out)
    desenet_branch_out = keras.layers.Conv2D(1024, 1, strides=1, padding='same')(densenet_out)

    concat = Concatenate()([resent_branch_out, desenet_branch_out])
    se_block_out = squeeze_excite_block(concat)
    
    x_F = tf.keras.layers.GlobalAveragePooling2D()(se_block_out)
    classification_fusion = tf.keras.layers.Dense(
        n_classes, activation="sigmoid", name='fusion_global')(x_F)
    model = tf.keras.models.Model(
        inputs=img_input, outputs= classification_fusion)
    return model