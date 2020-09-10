__author__ = 'Brian M Anderson'
# Created on 9/8/2020
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.losses import LossFunctionWrapper, losses_utils, ops, math_ops, array_ops, smart_cond
from tensorflow.python.keras.backend import nn, variables_module, variable, _constant_to_tensor, clip_ops, epsilon


class ExpandDimension(tf.keras.layers.Layer):
    def __init__(self, axis):
        super(ExpandDimension, self).__init__()
        self.axis = axis

    def call(self, input, **kwargs):
        return K.expand_dims(input, self.axis)


class SqueezeDimension(tf.keras.layers.Layer):
    def __init__(self, axis):
        super(SqueezeDimension, self).__init__()
        self.axis = axis

    def call(self, input, **kwargs):
        return K.squeeze(input, self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SqueezeDimension, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def BreakUpSqueezeDimensions(image, og_image):
    og_shape = tf.shape(og_image)
    filters = image.shape[-1]
    image = tf.reshape(image, [og_shape[0], og_shape[1], og_shape[2], og_shape[3], filters])
    return image


class SqueezeAxes(tf.keras.layers.Layer):
    def __init__(self):
        super(SqueezeAxes, self).__init__()

    def call(self, x, **kwargs):
        og_shape = tf.shape(x)
        x = tf.reshape(x, [og_shape[0] * og_shape[1], og_shape[2], og_shape[3], 1])
        return x

    def get_config(self):
        config = {}
        base_config = super(SqueezeAxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedCategoricalCrossentropy(LossFunctionWrapper):
    def __init__(self, weights, from_logits=False, label_smoothing=0, reduction=losses_utils.ReductionV2.AUTO,
                 name='weighted_categorical_crossentropy'):
        self.weights = weights
        weights = variable(weights)

        def weighted_keras_categorical_crossentropy(target, output, from_logits=False, axis=-1):
            target.shape.assert_is_compatible_with(output.shape)
            if from_logits:
                return nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output, axis=axis)
            if not isinstance(output, (ops.EagerTensor, variables_module.Variable)):
                while output.op.type == 'Identity':
                    output = output.op.inputs[0]
                if output.op.type == 'Softmax':
                    # When softmax activation function is used for output operation, we
                    # use logits from the softmax function directly to compute loss in order
                    # to prevent collapsing zero when training.
                    # See b/117284466
                    assert len(output.op.inputs) == 1
                    output = output.op.inputs[0]
                    return nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output, axis=axis)
            # scale preds so that the class probas of each sample sum to 1
            output = output / math_ops.reduce_sum(output, axis, True)
            # Compute cross entropy from probabilities.
            epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
            output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
            return -math_ops.reduce_sum(target * math_ops.log(output)*weights, axis)

        def weighted_categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
            y_pred = ops.convert_to_tensor_v2(y_pred)
            y_true = math_ops.cast(y_true, y_pred.dtype)
            label_smoothing = ops.convert_to_tensor_v2(label_smoothing, dtype=K.floatx())

            def _smooth_labels():
                num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
                return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

            y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels, lambda: y_true)

            return weighted_keras_categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

        super(WeightedCategoricalCrossentropy, self).__init__(weighted_categorical_crossentropy, name=name, reduction=reduction,
                                                              from_logits=from_logits, label_smoothing=label_smoothing)


if __name__ == '__main__':
    pass
