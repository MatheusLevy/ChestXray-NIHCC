import tensorflow.keras as keras
from tensorflow.keras.regularizers import l1, l2

def activation_block(x):
    x = keras.layers.Activation("gelu")(x)
    return keras.layers.BatchNormalization()(x)

# Patch Embedding
def conv_stem(x, filters: int, patch_size: int):
    x = keras.layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)

def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = keras.layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = keras.layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x

def get_conv_mixer(
    image_size=32, filters=256, depth=8, kernel_size=5, patch_size=2, num_classes=10
):
        """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
        The hyperparameter values are taken from the paper.
        """
        inputs = keras.Input((image_size, image_size, 1))

        # Extract patch embeddings.
        x = conv_stem(inputs, filters, patch_size)

        # ConvMixer blocks.
        for _ in range(depth):
            x = conv_mixer_block(x, filters, kernel_size)

        # Classification block.
        x = keras.layers.GlobalAvgPool2D()(x)
        outputs = keras.layers.Dense(num_classes, activation="sigmoid", kernel_regularizer=l2(0.01))(x)

        return keras.Model(inputs, outputs)