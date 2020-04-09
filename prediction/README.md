# Scripts

save_chrom_preds_checkpoint # requires model config - loads config and uses it to define model
save_chrom_preds_final_model # instead of loading model config from a config file just hardcodes the relevant model config

The pyBigwig error
RuntimeError: Received an error during file opening!
Can be due to an error in the expression within the call to open
e.g. here: the os.path.join bracket is in the wrong place, but pyBigWig catches the exception
bw = pyBigWig.open(os.path.join(outdir, "{}.bigwig".format(val_track), "w"))
oddly this seemed to be resolved by first constructing the path, then passing this to pybigwig (i.e. passing as first arg a string variable rather than an expression which would be expected to evaluate to a string)