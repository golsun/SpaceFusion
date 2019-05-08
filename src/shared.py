import os, random, sys, datetime, time, socket, io, h5py, argparse, shutil, io
try:
	import queue
except ImportError:
	import Queue as queue
import numpy as np
import scipy

"""
AUTHOR: Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""

SOS_token = '_SOS_'
EOS_token = '_EOS_'
UNK_token = '_UNK_'

hostname = socket.gethostname()
SKIP_VIS = hostname not in ['MININT-3LHNLKS', 'xiag-0228']
FIT_VERBOSE = 1

if not SKIP_VIS:
	import matplotlib.pyplot as plt

if hostname in ['MININT-3LHNLKS', 'xiag-0228']:
	DATA_PATH = "d:/data"
	OUT_PATH = 'out'
elif os.getenv('PT_DATA_DIR') is not None:
	DATA_PATH = os.getenv('PT_DATA_DIR') + '/data'
	OUT_PATH  = os.getenv('PT_OUTPUT_DIR') + '/out'
	FIT_VERBOSE = 0
else:
	DATA_PATH = 'data'
	OUT_PATH = 'out'

print('@'*20)
print('hostname:  %s'%hostname)
print('data_path: %s'%DATA_PATH)
print('out_path:  %s'%OUT_PATH)
print('@'*20)

DECODER_LOSS = 'categorical_crossentropy'

max_seq_len = 32
batch_size = 256
max_num_sample = None
max_num_token = None

def str2bool(s):
	return {'true':True, 'false':False}[s.lower()]

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='ref10new')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--token_embed_dim', type=int, default=128)
parser.add_argument('--rnn_units', type=int, default=128)
parser.add_argument('--encoder_depth', type=int, default=2)
parser.add_argument('--decoder_depth', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--skip_perc', type=float, default=0.)
parser.add_argument('--stddev', type=float, default=0.1)
parser.add_argument('--wt_S2T', type=float, default=10.)
parser.add_argument('--wt_S2S', type=float, default=10.)
parser.add_argument('--wt_T2T', type=float, default=10.)
parser.add_argument('--interp', type=str2bool, default='True')
parser.add_argument('--load_1_batch','-1', action='store_true')
parser.add_argument('--restore', default='')
parser.add_argument('--epoch_load', type=int, default=0)	# 0 means the latest
parser.add_argument('--max_src_len', type=int, default=90)	
parser.add_argument('--max_tgt_len', type=int, default=30)	
parser.add_argument('--fld_suffix', default='')	

seed = 9
random.seed(seed)
np.random.seed(seed)



def strmap(x):
	if isinstance(x, str):
		return x
	if int(x) == x:
		return '%i'%x
	return '%.4f'%x


def int2str(i):
	if i < 1000:
		return str(i)
	else:
		k = i/1000
		if int(k) == k:
			return '%ik'%k
		else:
			return '%.1fk'%k


def makedirs(fld):
	if not os.path.exists(fld):
		os.makedirs(fld)


def line_generator(path, i_max=None):
	f = io.open(path, 'r', encoding='utf-8', errors='ignore')
	i = 0
	for line in f:
		i += 1
		yield line.strip('\n')
		if i == i_max:
			break

def rand_latent(center, r, limit=True):
	if r == 0:
		return center

	units = center.shape[1]
	noise = np.random.normal(size=center.shape)
	r_raw = np.sqrt(np.sum(np.power(noise, 2)))
	sampled = center + noise/r_raw*r
	if limit:
		return np.minimum(1, np.maximum(-1, sampled))
	else:
		return sampled


if __name__ == '__main__':
	for line in line_generator('lg.txt', i_max=10):
		print('[%s]'%line)