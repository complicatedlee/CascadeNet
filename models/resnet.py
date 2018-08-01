# -*- coding: utf-8 -*-

"""
keras_resnet.models._2d
~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular two-dimensional residual models.
"""

import keras.backend
import keras.layers
import keras.models
import keras.regularizers

import custom_layers.blocks

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101']

def ResNet(inputs, blocks, block, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` object using the given block count.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_2d`)

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.blocks
        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> blocks = [2, 2, 2, 2]

        >>> block = keras_resnet.blocks.basic_2d

        >>> model = keras_resnet.models.ResNet(x, classes, blocks, block, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    # x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1")(inputs)    
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", name="conv1")(inputs)  
    x = keras.layers.BatchNormalization(epsilon=1.1e-5, axis=axis, name="bn_conv1")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)
    # x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(features, stage_id, block_id, numerical_name=(blocks[stage_id] > 6))(x)

        features *= 2

        outputs.append(x)

    if include_top:
        assert classes > 0

        x = keras.layers.GlobalAveragePooling2D(name="pool5")(x)
        x = keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

        return keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)
    else:
        # Else output each stages features
        return keras.models.Model(inputs=inputs, outputs=outputs, *args, **kwargs)


def resnet18(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the resnet18 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.resnet18(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [2, 2, 2, 2]

    return ResNet(inputs, blocks, block=custom_layers.blocks.basic_2d, include_top=include_top, classes=classes, *args, **kwargs)


def resnet34(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the resnet34 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.resnet34(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 6, 3]

    return ResNet(inputs, blocks, block=custom_layers.blocks.basic_2d, include_top=include_top, classes=classes, *args, **kwargs)


def resnet50(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the resnet50 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.resnet50(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 6, 3]

    return ResNet(inputs, blocks, block=custom_layers.blocks.bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)


def resnet101(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the resnet101 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.resnet101(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 4, 23, 3]

    return ResNet(inputs, blocks, block=custom_layers.blocks.bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)


def ResNet152(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet152 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet152(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 8, 36, 3]

    return ResNet(inputs, blocks, block=custom_layers.blocks.bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)


def ResNet200(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet200 specifications.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    :param blocks: the network’s residual architecture

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)

    Usage:

        >>> import keras_resnet.models

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.models.ResNet200(x, classes=classes)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if blocks is None:
        blocks = [3, 24, 36, 3]

    return ResNet(inputs, blocks, block=custom_layers.blocks.bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)
