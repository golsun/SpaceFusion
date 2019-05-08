from shared import *

"""
AUTHOR: Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""


def load_vocab(path):
	with io.open(path, encoding='utf-8') as f:
		lines = f.readlines()

	index2token = dict()
	token2index = dict()
	for i, line in enumerate(lines):
		token = line.strip('\n').strip()
		index2token[i + 1] = token 			# start from 1, so 0 reserved for PAD
		token2index[token] = i + 1

	assert(SOS_token in token2index)
	assert(EOS_token in token2index)
	assert(UNK_token in token2index)
	return index2token, token2index


class Dataset:

	def __init__(self, 
		fld_data, 
		max_src_len=90,
		max_tgt_len=30,
		):

		path_vocab = os.path.join(fld_data, 'vocab.txt')
		self.path_train = os.path.join(fld_data, 'train.num')
		self.path_vali = os.path.join(fld_data, 'vali.num')
		self.path_test = os.path.join(fld_data, 'test.num')

		# load token dictionary

		self.max_src_len = max_src_len
		self.max_tgt_len = max_tgt_len
		self.index2token, self.token2index = load_vocab(path_vocab)

		self.SOS = self.token2index[SOS_token]
		self.EOS = self.token2index[EOS_token]
		self.UNK = self.token2index[UNK_token]
		self.num_tokens = len(self.token2index)	# not including 0-th

		# load source-target pairs, tokenized
		
		self.reset('train')
		self.reset('vali')
		self.reset('test')


	def reset(self, task):
		self.generator = {
			'train': line_generator(self.path_train),
			'vali': line_generator(self.path_vali),
			'test': line_generator(self.path_test),
			}

	def skip(self, n):
		if n == 0:
			return
		print('skipping %.2fM lines...'%(n/1e6))
		m = 0
		for line in self.generator['train']:
			m += 1
			if m == n:
				break


	def seq2txt(self, seq):
		words = []
		for j in seq:
			if j == 0:		# skip PAD
				continue	
			words.append(self.index2token[int(j)])
		return ' '.join(words)


	def txt2seq(self, text):
		tokens = text.strip().split(' ')
		seq = []
		for token in tokens:
			seq.append(self.token2index.get(token, self.token2index[UNK_token]))
		return seq


	def load_data(self, task, max_n, check_src=False):
		print('loading data, check_src = %s...'%check_src)

		auto_encoder_input_data = np.zeros((max_n, self.max_tgt_len))
		dial_encoder_input_data = np.zeros((max_n, self.max_src_len))
		decoder_input_data = np.zeros((max_n, self.max_tgt_len + 2))
		decoder_target_data = np.zeros((max_n, self.max_tgt_len + 2, self.num_tokens + 1))	

		# shape = [sample, len, vocab]
		# len: +2 as will 1) add EOS and 2) shift to right by 1 time step
		# vocab: +1 as mask_zero (token_id == 0 means PAD)

		source_texts = []
		target_texts = []

		i = 0
		max_src_len = 0
		max_tgt_len = 0
		prev_src = ''
		for line in self.generator[task]:

			seq_source, seq_target = line.split('\t')
			if check_src and (seq_source == prev_src):
				continue
			prev_src = seq_source

			seq_source = [int(j) for j in seq_source.split()]
			seq_target = [int(j) for j in seq_target.split()]
			seq_source = seq_source[-min(len(seq_source), self.max_src_len):]
			seq_target = seq_target[-min(len(seq_target), self.max_tgt_len):]

			source_texts.append(self.seq2txt(seq_source))
			target_texts.append(self.seq2txt(seq_target))

			max_src_len = max(max_src_len, len(seq_source))

			# encoder
			for t, token_index in enumerate(seq_source):
				dial_encoder_input_data[i, t] = token_index
			for t, token_index in enumerate(seq_target):
				auto_encoder_input_data[i, t] = token_index
		
			# decoder
			for t, token_index in enumerate(seq_target):
				decoder_input_data[i, t + 1] = token_index				# shift 1 time step
				decoder_target_data[i, t, token_index] = 1.

			decoder_input_data[i, 0] = self.SOS 						# inp starts with EOS
			decoder_target_data[i, len(seq_target), self.EOS] = 1.		# out ends with EOS
			max_tgt_len = max(max_tgt_len, len(seq_target) + 1)			# +1 as EOS added

			i += 1
			if i == max_n:
				break

		auto_encoder_input_data = auto_encoder_input_data[:i, :max_tgt_len]
		dial_encoder_input_data = dial_encoder_input_data[:i, :max_src_len]
		decoder_input_data = decoder_input_data[:i, :max_tgt_len]
		decoder_target_data = decoder_target_data[:i, :max_tgt_len, :]

		encoder_input_data = {
			'auto': auto_encoder_input_data,
			'dial': dial_encoder_input_data
			}

		print('size: in = %s, out = %s'%(dial_encoder_input_data.shape, decoder_target_data.shape))
		return encoder_input_data, decoder_input_data, decoder_target_data, source_texts, target_texts
		
