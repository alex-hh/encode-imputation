import tensorflow as tf

from keras import backend as K
from keras.layers import Input, Conv1D, Embedding, Dropout, Flatten, Dense,\
                         Concatenate, Lambda
from keras.models import Model

from models.layers import CustomDropout

from utils.CONSTANTS import dataset_celltypes, dataset_assaytypes


class DAE:

  def __init__(self, obs_counts=[50,45], n_drop=50, n_layers=2, cell_embed_dim=32,
               assay_embed_dim=256, num_hidden=2048, dae_dim=25, mlp_dropout=0, n_train_obs=267):
    self.config = locals()
    del self.config['self']
    cell_ids = dataset_celltypes['train']
    assay_ids = dataset_assaytypes['train']
    cell_embed_dim = cell_embed_dim
    assay_embed_dim = assay_embed_dim
    n_layers = n_layers
    # does dropout mask a random number of units each time or the same ? does it do floor?

    cell_embedding = Embedding(len(cell_ids), cell_embed_dim, name="celltype_embedding")
    assay_embedding = Embedding(len(assay_ids), assay_embed_dim, name="assay_embedding")

    if type(dae_dim) == int:
      dae_dense = [Dense(dae_dim, activation='relu')]
    elif type(dae_dim) == list:
      dae_dense = [Dense(dd, activation='relu') for dd in dae_dim]

    mlp_layers = []
    for i in range(n_layers):
      mlp_layers.append(Conv1D(num_hidden, 1, activation='relu'))
    mlp_output = Conv1D(1,1,activation=None)
    
    self.models = {}
    for c in obs_counts:
      drop_ids = Input(shape=(n_drop,), dtype='int32', name='drop_inds')
      predict_ids = Input(shape=(c,), dtype='int32', name='predict_inds')
      all_cell_ids = Input(shape=(c,), name='cell_ids'.format(c)) # batch_size, n_obs (cell1 cell1, cell3 , cell4) if expts are (C01M01, C01M02, C03M11, C04M15)
      all_assay_ids = Input(shape=(c,), name='assay_ids'.format(c)) # batch_size, n_obs (m1, m2, m11, m15) in above example
      measurements = Input(shape=(n_train_obs,), name='y_obs')
      inputs = [all_cell_ids, all_assay_ids, measurements, drop_ids, predict_ids]

      obs_d = CustomDropout()([measurements, drop_ids])
      h_obs = obs_d
      for dd in dae_dense:
        h_obs = dd(h_obs) # (batch_size, 25)
      obs_emb_rep = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=1), c, 1),
                           output_shape=lambda input_shape: input_shape[:1] + (c,) + input_shape[1:])
      obs_rep = obs_emb_rep(h_obs)

      embeddings = [cell_embedding(all_cell_ids), assay_embedding(all_assay_ids), obs_rep]

      full_rep = Concatenate()(embeddings)

      x = full_rep
      for i in range(n_layers):
        x = mlp_layers[i](x)
        if mlp_dropout > 0:
          x = Dropout(mlp_dropout)(x) # noise_shape=(None, 1, None) should apply same mask to each measurement at a given position
      y = mlp_output(x)
      y = Flatten()(y)
      model = Model(inputs, y)
      self.models[c] = model