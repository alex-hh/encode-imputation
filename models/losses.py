import tensorflow as tf
import numpy as np

from keras import backend as K
from keras.losses import mean_squared_error
# from tensorflow.losses import Reduction # the reduction to apply to loss
# https://www.tensorflow.org/api_docs/python/tf/compat/v2/keras/losses/Reduction

from functools import partial, update_wrapper


# Keras losses: function that returns a scalar for each data-point and takes (y_true, y_pred)
# The actual optimized objective is the mean of the output array across all datapoints.

# Typically robust loss functions diverge from the mse above z = (y_true - y_pred)**2 = 1
# p(z) behaves differently for z > 1 than for z < 1 where it behaves similarly to p(z) = z
# To force the robust loss function to instead diverge at (y_true - y_pred)**2 = k
# scale - > z' = (y_true - y_pred)**2 / k**2
# e.g. if i want to penalize errors of size 5 (missing a 10^-5 peak) harshly
# but then be less harsh afterwards
# z'(5) = 5**2 / 5**2 = 1
# equivalent to scaling values by sqrt(k) I suppose

# I think the scale of the ERRORS at which I want to start robustifying is around 5-10
# an error y_true - y_pred = 5 means that we have got a p-value that is wrong by a factor of 100000
# most of the meaningful differences happen on this scale


def wrapped_partial(func, name=None, *args, **kwargs):
  partial_func = partial(func, *args, **kwargs)
  update_wrapper(partial_func, func)
  if name is not None:
    partial_func.__name__ = name
  return partial_func

# def huber_loss(y_true, y_pred, delta=1.0):
#   return tf.losses.huber_loss(y_true, y_pred, delta=delta, reduction=Reduction.NONE)

def cauchy_loss(y_true, y_pred):
  return K.mean(K.log(1.0+K.square(y_pred - y_true)), axis=-1) # (batch_size, 1)

def robust_squared_error(y_true, y_pred, robustifier=lambda x: x, scale=1.0):
  scaled_squared_error = K.square((y_pred - y_true)/scale)
  # http://ceres-solver.org/nnls_modeling.html?highlight=loss%20function#instances
  return K.mean((scale**2)*robustifier(scaled_squared_error), axis=-1)

def cauchy_robustifier(z):
  return K.log(1.0 + z)

def huber_robustifier(z):
    """
    https://github.com/tensorflow/tensorflow/blob/456fbc0e498e3d10604973de9f46ca48d62267cc/tensorflow/python/ops/losses/losses_impl.py#L422
    https://stackoverflow.com/questions/47840527/using-tensorflow-huber-loss-in-keras
    For each value x in error=labels-predictions, the following is calculated:

     0.5 * x^2                  if |x| <= d
     0.5 * d^2 + d * (|x| - d)  if |x| > d
    N.B. that if d=1 this simplifies to 0.5 * 1 + 1*abs(y_true-y_pred) - 1 = abs(y_true-y_pred) - 0.5
    """
    return tf.where(z<1.0, 0.5*z, K.sqrt(z) - 0.5)

def logcosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.
    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction.
    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
    # Returns
        Tensor with one scalar loss entry per sample.
    """
    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)
    return K.mean(_logcosh(y_pred - y_true), axis=-1)

# huber2 = wrapped_partial(huber_loss, delta=2.0)
# huber5 = wrapped_partial(huber_loss, delta=5.0) # when y = 5 this means p = 10^-5. for p > 10^-5
# huber10 = wrapped_partial(huber_loss, delta=10.0)
# huber25 = wrapped_partial(huber_loss, delta=25.0)
# huber50 = wrapped_partial(huber_loss, delta=50.0)
# huber100 = wrapped_partial(huber_loss, delta=100.0)

# huber2r = wrapped_partial(robust_squared_error, robustifier=huber_robustifier, delta=2.0)
# huber5r = wrapped_partial(robust_squared_error, robustifier=huber_robustifier, delta=5.0) # when y = 5 this means p = 10^-5. for p > 10^-5
# huber10r = wrapped_partial(robust_squared_error, robustifier=huber_robustifier, delta=10.0)
# huber25r = wrapped_partial(robust_squared_error, robustifier=huber_robustifier, delta=25.0)
# huber50r = wrapped_partial(robust_squared_error, robustifier=huber_robustifier, delta=50.0)
# huber100r = wrapped_partial(robust_squared_error, robustifier=huber_robustifier, delta=100.0)

cauchy1 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=1.0)
cauchy2 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=2.0)
cauchy3 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=3.0)
cauchy4 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=4.0)
cauchy5 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=5.0)
cauchy6 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=6.0)
cauchy7 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=7.0)
cauchy8 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=8.0)
cauchy9 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=9.0)
cauchy10 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=10.0)
cauchy25 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=25.0)
cauchy50 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=50.0)
cauchy100 = wrapped_partial(robust_squared_error, robustifier=cauchy_robustifier, scale=100.0)

# Functions for computing the raw mse from transformed y_true, y_pred (e.g. undo arcsinh)
def _inv_transformed_mse(y_true, y_pred, inv_transform_func=np.sinh, scale=1.0, center=0.0):
  return mean_squared_error(inv_transform_func((y_true*scale)+center),
                            inv_transform_func((y_pred*scale)+center))

def inv_transformed_mse(undo_transform='arcsinh', scaler=None, expt_names=[]):
  assert undo_transform in [None, 'arcsinh', 'sqrt']
  if undo_transform == 'arcsinh':
    inv_transform_func = tf.sinh
  elif undo_transform == 'sqrt':
    inv_transform_func = K.square
  elif undo_transform is None:
    inv_transform_func = K.identity
  mapped_scale_ = 1.0
  mapped_center_ = 0.0
  if scaler is not None:
    mapped_scale_ = np.asarray([scaler.grouped_scale_[e[3:]] for e in expt_names])
    mapped_center_ = np.asarray([scaler.grouped_center_[e[3:]] for e in expt_names])
  return wrapped_partial(_inv_transformed_mse, name='mse_raw', inv_transform_func=inv_transform_func,
                         scale=mapped_scale_, center=mapped_center_)

sinh_mse = wrapped_partial(_inv_transformed_mse, inv_transform_func=tf.sinh)
squared_mse = wrapped_partial(_inv_transformed_mse, inv_transform_func=K.square)
