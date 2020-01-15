import time, glob, os

class timeit:

    def __enter__(self):
        self.t1 = time.time()

    def __exit__(self, *args):
        print('Executed in:\t{}s'.format(time.time()-self.t1), flush=True)

for f in glob.glob('data/evaluation_data/avocado/*.npz'):
    expt = f.split('/')[-1].split('.')[0]
    # chrom = f.split('/')[-1].split('.')[1]
    chrom = 'chr21'
    os.rename(f, 'data/evaluation_data/avocado/{}.{}.gz.npz'.format(expt, chrom))