import tensorflow as tf
from keras.layers import Layer
from keras import backend as K

class CustomDropout(Layer):
  # def build(self, max_batch_size):
  #   self.mask = K.ones((10000,))

  def call(self, x, training=None):
    # alternative to this is to pass in a mask 
    # https://stackoverflow.com/questions/40443951/binary-mask-in-tensorflow/40444239#40444239
    # https://stackoverflow.com/questions/49755316/best-way-to-mimic-pytorch-sliced-assignment-with-keras-tensorflow
    # https://stackoverflow.com/questions/45162998/proper-usage-of-tf-scatter-nd-in-tensorflow-r1-2
    # https://stackoverflow.com/questions/47338139/slice-tensor-with-variable-indexes-with-lambda-layer-in-keras

    # if we wanted to use scatter
    # the indices we would want would be
    # ([0, row0[0]], [0, row0[1], ...]), etc.
    # pretty horrible 

    inputs, mask_inds = x[0], x[1]
    # self.mask = tf.Variable(K.ones((10000,inputs.shape[1])))
    # self.mask = tf.Variable(K.ones((K.shape(inputs)[0], K.shape(inputs)[1])), validate_shape=False) # it doesn't like initialising a variable without knowing the shape

    def dropped_inputs():
      # https://stackoverflow.com/questions/48515034/keras-tensorflow-initializer-for-variable-is-from-inside-a-control-flow-con?rq=1
      # mask = tf.Variable(initial_value=lambda: K.ones((10,5), dtype=tf.float32), validate_shape=False)
      # mask = K.variable(K.ones_like(inputs))
      base_inds = tf.tile(tf.range(tf.shape(inputs)[0])[:,tf.newaxis], (1, tf.shape(mask_inds)[1]))
      selection_array = tf.stack([base_inds, mask_inds], axis=-1)
      # mask = tf.scatter_nd_sub(self.mask, selection_array, K.ones_like(mask_inds, dtype=tf.float32))
      mask = tf.scatter_nd(selection_array, K.ones_like(mask_inds), tf.shape(inputs))
      mask = tf.equal(mask, 0) # boolean array - true for indices not in mask inds
      mask = tf.cast(mask, tf.float32)
      # tf.shape(mask_inds)[1] b.c. mask_inds has a batch dim too
      keep_prob = (tf.shape(inputs)[1] - tf.shape(mask_inds)[1])/tf.shape(inputs)[1]
      scale = 1 / keep_prob
      ret = inputs * tf.cast(scale, tf.float32) * mask[:tf.shape(inputs)[0],:tf.shape(inputs)[1]]
      return ret
    return K.in_train_phase(dropped_inputs, inputs,
                            training=training)
    return inputs

  def compute_output_shape(self, input_shape):
    return input_shape

class CustomSeqDropout(Layer):
  # def build(self, max_batch_size):
  #   self.mask = K.ones((10000,))

  def call(self, x, training=None):
    """
    Will be fed an input tensor and a list of inds which should be zeroed out in the last axis of the tensor
    at each timestep

    N.B. we could handle this in the dataloader, but there are certain advantages to doing it in keras
    In particular the unmasked values should be scaled during training and not during testing
    And keras handles this automatically
    (Although handling it myself would frankly give me greater control)
    """
    inputs, mask_inds = x[0], x[1] # (batch_size, sequence_len, n_obs) and (batch_size, n_drop)
    # self.mask = tf.Variable(K.ones((10000,inputs.shape[1])))
    # self.mask = tf.Variable(K.ones((K.shape(inputs)[0], K.shape(inputs)[1])), validate_shape=False) # it doesn't like initialising a variable without knowing the shape

    def dropped_inputs():
      # https://stackoverflow.com/questions/45162998/proper-usage-of-tf-scatter-nd-in-tensorflow-r1-2
      print('mask inds', mask_inds)
      batch_size, sequence_len, n_obs = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
      nmask_inds = tf.tile(mask_inds[:,tf.newaxis, :], [1, sequence_len, 1])
      i1, i2 = tf.meshgrid(tf.range(batch_size),
                           tf.range(sequence_len), indexing="ij")
      i1 = tf.tile(i1[:, :, tf.newaxis], [1, 1, tf.shape(nmask_inds)[-1]])
      i2 = tf.tile(i2[:, :, tf.newaxis], [1, 1, tf.shape(nmask_inds)[-1]])
      # Create final indices
      selection_array = tf.stack([i1, i2, nmask_inds], axis=-1)
      # Output shape
      # to_shape = [batch_size, sequence_len, vocab_size]
      # Get scattered tensor
      mask = tf.scatter_nd(selection_array, K.ones_like(nmask_inds), tf.shape(inputs))
      # right, now invert the mask actually
      mask = tf.equal(mask, 0) # boolean array - true for indices not in mask inds
      mask = tf.cast(mask, tf.float32)
      keep_prob = (tf.shape(inputs)[-1] - tf.shape(mask_inds)[-1])/tf.shape(inputs)[-1]
      scale = 1 / keep_prob
      ret = inputs * tf.cast(scale, tf.float32) * mask
      return ret
    return K.in_train_phase(dropped_inputs, inputs,
                            training=training)
    return inputs

  def compute_output_shape(self, input_shape):
    return input_shape

class CustomDropoutV2(Layer):
  def __init__(self,  **kwargs):
    super(Dropout, self).__init__(**kwargs)

  def call(self, inputs, mask, drop=2, mask_axis=1, training=None):
    # alternative to this is to pass in a mask 
    # https://stackoverflow.com/questions/40443951/binary-mask-in-tensorflow/40444239#40444239
    # https://stackoverflow.com/questions/49755316/best-way-to-mimic-pytorch-sliced-assignment-with-keras-tensorflow
    # https://stackoverflow.com/questions/45162998/proper-usage-of-tf-scatter-nd-in-tensorflow-r1-2
    # https://stackoverflow.com/questions/47338139/slice-tensor-with-variable-indexes-with-lambda-layer-in-keras

    # if we wanted to use scatter
    # the indices we would want would be
    # ([0, row0[0]], [0, row0[1], ...]), etc.
    # pretty horrible 
    def dropped_inputs():
      n_drop = K.sum(mask) / tf.shape(mask)[0]
      keep_prob = (tf.shape(inputs)[1] - n_drop)/tf.shape(inputs)[1]
      scale = 1 / keep_prob
      ret = inputs * scale * mask
      return ret
    return K.in_train_phase(dropped_inputs, inputs,
                            training=training)

  def compute_output_shape(self, input_shape):
    return input_shape