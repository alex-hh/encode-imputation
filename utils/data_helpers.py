import numpy as np


def apply_transform(vals, transform=None):
  assert transform in [None, 'arcsinh', 'sqrt', 'cbrt', '2.5rt', '1.5rt']
  if transform == 'arcsinh':
    vals = np.arcsinh(vals)
  elif transform == 'sqrt':
    vals = np.sqrt(vals)
  elif transform == 'cbrt':
    vals = np.cbrt(vals)
  elif transform == '2.5rt':
    vals = np.power(vals,2/5) # 2/5 = 1/2.5 i.e. 2.5th root
  elif transform == '1.5rt':
    vals = np.power(vals,2/3)
  return vals

def get_inverse_transform_func(transform=None):
  assert transform in [None, 'arcsinh', 'sqrt', 'cbrt', '2.5rt', '1.5rt']
  if transform == 'arcsinh':
    return np.sinh
  if transform == 'sqrt':
    return np.square
  if transform == None:
    return lambda x: x

def inverse_transform(vals, transform=None):
  assert transform in [None, 'arcsinh', 'sqrt', 'cbrt', '2.5rt', '1.5rt']
  if transform == 'arcsinh':
    vals = np.sinh(vals)
  elif transform == 'sqrt':
    vals = np.square(vals)
  elif transform == 'cbrt':
    vals = np.power(vals, 3)
  elif transform == '2.5rt':
    vals = np.power(vals, 2.5) # 2/5 = 1/2.5 i.e. 2.5th root
  elif transform == '1.5rt':
    vals = np.power(vals, 1.5)
  return vals