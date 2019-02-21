# LBS

LBS is a general framework for screening in high dimensional data. It can take advantage of multiple CPU cores to significantly accelerate the running time.


## Example

## Training a screening model by LBS on the HIV-1 integrase dataset:

python lbsTrain.py --data 'data/integrase.csv.gz' --out lbsmodel.npz


## Screening samples on test data :

python lbsPredict.py --data 'data/integrase.csv.gz' --model lbsmodel.npz --result SreenResult.txt


## Usage

lbsTrain.py

Training by LBS (Local Beta Screening)

usage: lbsTrain.py [-h] [--data D] [--ne N] [--out O] [--cpu C]

optional arguments:
  -h, --help  show this help message and exit
  --data D    data to be trained, the last column is regarded as label
              (required)
  --ne N      maximum number of feature subsets to be evaluated in each
              iteration. the default is 2000000 (optional)
  --out O     filename to keep the output of training. the default is
              lbsmodel.npz (optional)
  --cpu C     the number of CPUs to use. the default is to use all of CPUs
              available (optional)


lbsPredict.py

Screening by trained LBS (Local Beta Screening)

usage: lbsPredict.py [-h] [--data D] [--model M] [--result R]
optional arguments:
  -h, --help  show this help message and exit
  --data D    data to be screened (required)
  --model M   filename of the LBS model trained. the default is lbsmodel.npz
              (optional)
  --result R  filename to keep the result of screening, index of screened
              samples are kept. the default is ScreenResult.txt (optional)

