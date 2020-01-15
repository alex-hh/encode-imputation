from functools import partial, update_wrapper

import numpy as np
import tensorflow as tf

from keras import backend as K
from scipy.stats import spearmanr


def evaluate_predictions(unfiltered_preds, unfiltered_targets, gene_bins=None,
                         enhancer_bins=None, promoter_bins=None, retain_bins=None):
    """
    given a file of predictions and a file of targets compute evaluation metrics and return dict containing results
    (including a sum of squared errors so that errors can be added across multiple chromosomes)

    metrics can be computed on gene/enhancer/promoter subsets by passing lists of bin ids matching these subsets
            (optionally after filtering by eg. presence on blacklist via interval_filter)
    """
    unfiltered_errors = np.square(unfiltered_preds-unfiltered_targets)
    if retain_bins is None:
        retain_bins = np.arange(unfiltered_errors.shape[0])
    # print('unfiltered errors shape, unfiltered errors size', unfiltered_errors.shape, unfiltered_errors.size)

    gene_weights = np.zeros(unfiltered_preds.shape) # just (binned_chrom_length,)
    enh_weights = np.zeros(unfiltered_preds.shape)
    prom_weights = np.zeros(unfiltered_preds.shape)

    sse_dict = {}
    count_dict = {} # store number of bins we're summing over for each metric

    corr = gwcorr(unfiltered_targets, unfiltered_preds)
    spear = gwspear(unfiltered_targets, unfiltered_preds)

    sse_dict['cwcorr'] = corr
    sse_dict['cwspear'] = spear
    count_dict['cwcorr'] = 1
    count_dict['cwspear'] = 1

    if gene_bins is not None:
      gene_weights[gene_bins] = 1.0
      gene_errors = gene_weights*unfiltered_errors
      gene_errors = gene_errors[retain_bins].sum()
      sse_dict['gene'] = gene_errors
      count_dict['gene'] = np.sum(gene_weights[retain_bins])

    if promoter_bins is not None:
      prom_weights[promoter_bins] = 1.0
      prom_errors = prom_weights*unfiltered_errors
      prom_errors = prom_errors[retain_bins].sum()
      sse_dict['prom'] = prom_errors
      count_dict['prom'] = np.sum(prom_weights[retain_bins])

    if enhancer_bins is not None:
      enh_weights[enhancer_bins] = 1.0
      enh_errors = enh_weights*unfiltered_errors
      enh_errors = enh_errors[retain_bins].sum()
      sse_dict['enh'] = enh_errors
      count_dict['enh'] = np.sum(enh_weights[retain_bins])

    errors = unfiltered_errors[retain_bins]

    targets = unfiltered_targets[retain_bins]
    assert targets.shape == errors.shape
    del unfiltered_targets
    # TODO - use np.percentile
    n_1pc = int(targets.shape[0] * 0.01)
    y_true_sorted = np.sort(targets)
    y_true_top1 = y_true_sorted[-n_1pc]
    # y_true_top1 = np.percentile(targets, 99)
    # print(y_true_top1, np.percentile(targets, 99))
    idx_obs = targets >= y_true_top1
    del targets
    top1_obs = errors[idx_obs]
    count_dict['top1_obs'] = top1_obs.size #.shape[0] would be exactly the same
    sse_dict['top1_obs'] = top1_obs.sum()

    preds = unfiltered_preds[retain_bins]
    assert preds.shape == errors.shape
    del unfiltered_preds
    n_1pc = int(preds.shape[0] * 0.01)
    y_pred_sorted = np.sort(preds)
    y_pred_top1 = y_pred_sorted[-n_1pc]
    idx_pred = preds >= y_pred_top1
    del preds
    top1_preds = errors[idx_pred]
    count_dict['top1_preds'] = top1_preds.size
    sse_dict['top1_preds'] = top1_preds.sum()

    count_dict['global'] = errors.size
    sse_dict['global'] = errors.sum()

    return sse_dict, count_dict

def wrapped_partial(func, name=None, *args, **kwargs):
  partial_func = partial(func, *args, **kwargs)
  update_wrapper(partial_func, func)
  if name is not None:
    partial_func.__name__ = name
  return partial_func


# import tensorflow as tf
def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]

# def correlation_coefficient(y_true, y_pred):
#     pearson_r, update_op = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true, name='pearson_r')
#     return pearson_r

# https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras/46620771
def correlation_coefficient_up(y_true, y_pred):
    pearson_r, update_op = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true, name='pearson_r')
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'pearson_r'  in i.name.split('/')]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        pearson_r = tf.identity(pearson_r)
    return update_op

def correlation_coefficient_r(y_true, y_pred):
    pearson_r, update_op = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true, name='pearson_r')
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'pearson_r'  in i.name.split('/')]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        pearson_r = tf.identity(pearson_r)
    return pearson_r

def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r

def correlation_coefficient_loss_rowwise(y_true, y_pred):
    """
     N.B. be very careful with this...it simply can't work for batched evaluation can it?
    """
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym), axis=0)
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm), axis=0), K.sum(K.square(ym), axis=0)))
    r = r_num / r_den

    # r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r

def corr_slice(y_true, y_pred, slice_inds=[]):
    return K.mean(correlation_coefficient_loss_rowwise(tf.gather(y_true, slice_inds, axis=-1), tf.gather(y_pred, slice_inds, axis=-1)), axis=-1)

def subset_corr(inds, name):
    return wrapped_partial(corr_slice, name=name, slice_inds=inds)

def subset_mse(inds, name, undo_transform=None, scaler=None, expt_names=[]):
    assert undo_transform in ['arcsinh', 'sqrt', None]
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
    return wrapped_partial(mse_slice, name=name, slice_inds=inds, inv_transform_func=inv_transform_func,
                           scale=mapped_scale_, center=mapped_center_)

def mse_slice(y_true, y_pred, slice_inds=[], inv_transform_func=K.identity, scale=None, center=None):
    return K.mean(K.square(tf.gather(inv_transform_func((y_pred*scale)+center), slice_inds, axis=-1) - tf.gather(inv_transform_func((y_true*scale)+center), slice_inds, axis=-1)), axis=-1)

## THESE ARE ALL APPLIED ON A PER-TRACK BASIS
def np_mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()


def gwcorr(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def gwspear(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


def mseprom(y_true_dict, y_pred_dict, chroms,
            gene_annotations,
            window_size=25, prom_loc=80):
    """
    Args:
        y_true_dict: truth vector per chromosome
            { chr: y_true } where y_true is a np 1-dim array.
        y_pre_dict: predicted vector per chromosome
            { chr: y_pred } where y_pred is a np 1-dim array.
    """
    sse, n = 0., 0.

    for chrom in chroms:
        y_true = y_true_dict[chrom]
        y_pred = y_pred_dict[chrom]

        for line in gene_annotations:
            chrom_, start, end, _, _, strand = line.split()
            start = int(start) // window_size
            end = int(end) // window_size + 1

            # if chrom_ in ('chrX', 'chrY', 'chrM'):
            #     continue

            if chrom_ != chrom:
                continue

            if strand == '+':
                sse += ((y_true[start-prom_loc: start] -
                         y_pred[start-prom_loc: start]) ** 2).sum()
                n += y_true[start-prom_loc: start].shape[0]

            else:
                sse += ((y_true[end: end+prom_loc] -
                         y_pred[end: end+prom_loc]) ** 2).sum()
                n += y_true[end: end+prom_loc].shape[0]

    return sse / n

def msegene(y_true_dict, y_pred_dict, chroms,
            gene_annotations,
            window_size=25):
    sse, n = 0., 0.

    for chrom in chroms:
        y_true = y_true_dict[chrom]
        y_pred = y_pred_dict[chrom]

        for line in gene_annotations:
            chrom_, start, end, _, _, strand = line.split()
            start = int(start) // window_size
            end = int(end) // window_size + 1

            # if chrom_ in ('chrX', 'chrY', 'chrM'):
            #     continue

            if chrom_ != chrom:
                continue
            sse += ((y_true[start:end] - y_pred[start:end]) ** 2).sum()
            n += end - start

    return sse / n


def mseenh(y_true_dict, y_pred_dict, chroms,
           enh_annotations,
           window_size=25):
    sse, n = 0., 0.

    for chrom in chroms:
        y_true = y_true_dict[chrom]
        y_pred = y_pred_dict[chrom]

        for line in enh_annotations:
            chrom_, start, end, _, _, _, _, _, _, _, _, _ = line.split()
            start = int(start) // window_size
            end = int(end) // window_size + 1

            if chrom_ != chrom:
                continue
            sse += ((y_true[start:end] - y_pred[start:end]) ** 2).sum()
            n += end - start

    return sse / n


def msevar(y_true, y_pred, y_all=None, var=None):
    """Calculates the MSE weighted by the cross-cell-type variance.
    According to the wiki: Computing this measure involves computing,
    for an assay carried out in cell type x and assay type y, a vector of
    variance values across all assays of type y. The squared error between
    the predicted and true value at each genomic position is multiplied by
    this variance (normalized to sum to 1 across all bins) before being
    averaged across the genome.
    Parameters
    ----------
    y_true: np.ndarray, shape=(n_positions,)
        The true signal
    y_pred: np.ndarray, shape=(n_positions,)
        The predicted signal
    y_all: np.ndarray, shape=(n_celltypes, n_positions)
        The true signal from all the cell types to calculate the variance over.
        mutually exclusive with var
    var: np.ndarray, shape=(n_positions,)
        pre-computed var vector
        mutually exclusive with y_all
    Returns
    -------
    mse: float
        The mean-squared error that's weighted at each position by the
        variance.
    """

    if var is None and y_all is None:
        return 0.0
    if var is None:
        var = np.std(y_all, axis=0) ** 2
    return ((y_true - y_pred) ** 2).dot(var)/var.sum()


def mse1obs(y_true, y_pred):
    n = int(y_true.shape[0] * 0.01)
    y_true_sorted = np.sort(y_true)
    y_true_top1 = y_true_sorted[-n]
    idx = y_true >= y_true_top1

    return mse(y_true[idx], y_pred[idx])


def mse1imp(y_true, y_pred):
    n = int(y_true.shape[0] * 0.01)
    y_pred_sorted = np.sort(y_pred)
    y_pred_top1 = y_pred_sorted[-n]
    idx = y_pred >= y_pred_top1
    return mse(y_true[idx], y_pred[idx])


def build_var_dict(npys, chroms):
    y_all = {}
    for c in chroms:
        y_all[c] = []

    for f in npys:
        y_dict = load_npy(f)
        y_dict_norm = normalize_dict(y_dict, chroms) # doesn't do anything - there was going to be a scaling here

        for c in chroms:
            y_all[c].append(y_dict_norm[c])

    var = {}
    for c in chroms:
        var[c] = np.std(np.array(y_all[c]), axis=0) ** 2

    return var

def find_robust_min_max(x, pct_thresh=0.05, top_bottom_bin_range=2000000):
    """
     Assumption is that bins which when sorted lie outside top_bottom_bin_range are outliers
     and the relevant 'robust' minimum and maximum are percentiles within the top_bottom_bin_range
     This is a bit lie the scikit-learn robust scaler idea; though not exactly
    """
    y = x[x > 0]
    idxs = np.argsort(y)
    abs_max = y[idxs[-1]]
    abs_min = y[idxs[0]]
    robust_max = y[idxs[-int(pct_thresh * top_bottom_bin_range)]]
    robust_min = y[idxs[int(pct_thresh * top_bottom_bin_range)]]
    log.info('Array length original, non-zero: {}, {}'.format(len(x), len(y)))
    log.info('Absolute min, max: {}, {}'.format(abs_min, abs_max))
    log.info('Robust min, max: {}, {}'.format(robust_min, robust_max))
    return robust_min, robust_max