# Training a model on a single chromosome


## Preprocessing


We worked with the data in hdf5 format, with all training observations for a single chromosome in a single h5 file.

To build these files for a single chromosome (e.g. chr21), run: 

  ``` bash
      python preprocessing/build_h5_from_bigwigs.py --directory /path/to/data/dir --chroms chr21
  ```

## Predictions

### Predictions for a single model

To compute predictions for a single model on a single chromosome (e.g. chr21), ensure that the saved model checkpoints are in a folder /path/to/output_dir/weights. The checkpoints files themselvesshould follow the naming convention used in the training callback to save weights, that is: \<model\_name\>\_ep\<checkpoint_number\>-\<loss\>.hdf5 (cf the construction of the base checkpoint path in training.expt_config_loaders.get_validation_callbacks) Then run:

  ``` bash
      python prediction/build_prediction_single.py model_name --train_dataset 'train' --dataset 'val' \
              --output_directory /path/to/output/dir --data_directory /path/to/data/dir
  ```

--train_dataset specifies the subset of tracks that the model was trained on (i.e. either just tracks included in the competition training set: 'train', or tracks included in both the training and validation sets: 'all')

--dataset specifies the subset of tracks on which to make predictions (i.e. either the validation set tracks: 'val', or the test set tracks: 'test')

--data_directory and --output_directory specify the directories in which the training hdf5 files and the model checkpoints can be found, respectively (the checkpoints should be within a weights/ subdirectory within output_directory, whereas the training data should be directly within data_directory). N.B. that instead of specifying the data and output directories via command line args they can instead be specified via the environment variables DATA_DIR and OUTPUT_DIR. Command line args will override the environment variables. If neither are specified, the directories will fall back to defaults (data/ and outputs/) within the path where the code is being run.

 
## Training

We trained models initialized with different random seeds on each chromosome
To train a single model on a single chromosome (e.g. chr21), run 

  ``` bash
      python training/chunked_train_single_chrom.py --train_dataset 'train' --chrom 'chr21' \
             --data_directory '/path/to/data/dir' \
             --output_directory '/path/to/output/dir'
  ```

--train_dataset specifies the subset of tracks to train on (i.e. either just tracks included in the competition training set: 'train', or tracks included in both the training and validation sets: 'all')

A model name can optionally be specified via the --model_name argument. This determines the names of the model checkpoints that are saved during training (see above). If model_name is not passed it will default to \<chrom\>_\<train_dataset\>.

## Requirements

Developed and run with Python 3.6, and keras with tensorflow backend. Tested with tensorflow 1.14. See requirements.txt for details of packages.
