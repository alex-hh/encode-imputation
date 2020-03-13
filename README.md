## Preprocessing

We worked with the data in hdf5 format, with all training observations for a single chromosome in a single h5 file.

To build these files run 

  ``` bash
      python preprocessing/build_h5_from_bigwigs.py --directory /path/to/data/dir --all_chroms
  ```

## Predictions

Given a set of saved weights (i.e. as a result of training the same model on each chromosome as outlined above), predictions were generated by computing predictions from each set of weights and averaging them.

The full pipeline can be performed by running prediction/build_prediction_ensemble.py.

As a starting point, to do this for a single chromosome using the weights that were used for the best team imp submission:


  ``` bash
      python prediction/build_prediction_ensemble.py imp -chrom_list chr21 --data_directory /path/to/data/dir
  ```

This assumes that the weights are in a subdirectory named imp within the data directory (i.e. the weights tarball, which contains the imp subdirectory, should be extracted within /path/to/data/dir, which should be wherever the h5 files of training data have been built.)

 
## Training

We trained models initialized with different random seeds on each chromosome
To train on a single chromosome (e.g. chr21), run 

  ``` bash
      python training/chunked_train_single_chrom.py --chrom chr21
  ```

## Requirements

Developed and run with Python 3.6. See requirements.txt for details of packages.