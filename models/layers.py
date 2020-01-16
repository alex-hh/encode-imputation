import tensorflow as tf
from keras.layers import Layer
from keras import backend as K


class CustomDropout(Layer):

  def call(self, x, training=None):
    
    inputs, mask_inds = x[0], x[1]
    
    def dropped_inputs():
      base_inds = tf.tile(tf.range(tf.shape(inputs)[0])[:,tf.newaxis], (1, tf.shape(mask_inds)[1]))
      selection_array = tf.stack([base_inds, mask_inds], axis=-1)
      mask = tf.scatter_nd(selection_array, K.ones_like(mask_inds), tf.shape(inputs))
      mask = tf.equal(mask, 0) # boolean array - true for indices not in mask inds
      mask = tf.cast(mask, tf.float32)
      keep_prob = (tf.shape(inputs)[1] - tf.shape(mask_inds)[1])/tf.shape(inputs)[1]
      scale = 1 / keep_prob
      ret = inputs * tf.cast(scale, tf.float32) * mask[:tf.shape(inputs)[0],:tf.shape(inputs)[1]]
      return ret
    return K.in_train_phase(dropped_inputs, inputs,
                            training=training)
    return inputs

  def compute_output_shape(self, input_shape):
    return input_shape
