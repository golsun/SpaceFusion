from shared import *
from tf_lib import *
from dataset import *
from model import *


"""
AUTHOR: Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""


def run_s2s(name, mode, args):
	# name = mtask, muli-task model and SpaceFusion model
	# name = dial, vanilla seq-to-seq model

	fld_data, fld_model, subfld = get_model_fld(args)
	dataset = Dataset(fld_data, max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
	path_word2vec = None

	fld = os.path.join(fld_model, subfld)
	if name == 'mtask':
		s2s = MTask(
				dataset, 
				fld, 
				args.token_embed_dim, 
				args.rnn_units, 
				args.encoder_depth, 
				args.decoder_depth,
				dropout_rate=args.dropout,
				stddev=args.stddev,
				wt_S2T=args.wt_S2T,
				wt_S2S=args.wt_S2S,
				wt_T2T=args.wt_T2T,
				interp=args.interp,
				lr=args.lr,
				new=(mode=='train'),
				path_word2vec=path_word2vec)

	elif name in ['auto', 'dial']:
		s2s = Seq2Seq(
				dataset, 
				fld, 
				args.token_embed_dim, 
				args.rnn_units, 
				args.encoder_depth, 
				args.decoder_depth,
				dropout_rate=args.dropout,
				prefix=name,
				lr=args.lr,
				new=(mode=='train'),
				path_word2vec=path_word2vec)
	else:
		raise ValueError

	if mode not in ['train', 'summary']:
		if args.epoch_load == 0:
			restore_path = args.restore
		else:
			assert(args.restore == '')
			restore_path = s2s.fld + '/epochs/%s_epoch%i.weight.h5'%(s2s.model_name, args.epoch_load)
		s2s.load_weights(restore_path)

	if mode == 'load':
		return s2s

	print('\n'+fld+'\n')
	if mode == 'vali':
		loss_vali = s2s.vali()
		ss = [
			'd(S,T):%.4f'%loss_vali[1],
			'd(S,S):%.4f'%loss_vali[2],
			'd(T,T):%.4f'%loss_vali[3],
			'auto:  %.4f'%loss_vali[4],
			'dial:  %.4f'%loss_vali[5],
			]
		if s2s.interp:
			ss.append('interp:%.4f'%loss_vali[-1])
		print('\n'.join(ss))

	elif mode in ['continue', 'train']:
		if args.load_1_batch == 1:
			batch_per_load = 1
		else:
			batch_per_load = 50
		s2s.train(batch_size, args.epochs, batch_per_load, skip=args.skip, epoch_init=args.epoch_load)

	elif 'interact' in mode:
		beam_width = None
		if '_' not in mode:
			method = 'greedy'
		else:
			param = mode.split('_')[1]
			if param != 'rand':
				method = 'beam'
				beam_width = int(param)
			else:
				method = 'rand'
		print(method)

		while True:
			print('\n---- please input ----')
			input_text = input()
			if input_text == '':
				break
			results = s2s.dialog(input_text, prefix='dial', method=method, beam_width=beam_width)
			if method != 'beam':
				results = [results]
			for result in results:
				print('%.2f'%result[0] + '\t' + result[1])

	elif mode == 'test':
		r = 1.5
		lines = []
		prev = None
		if args.path_test == '':
			path_test = fld_data + '/test.txt'
		else:
			path_test = args.path_test
		for line in open(path_test, encoding='utf-8'):
			src, _ = line.strip('\n').split('\t')
			if src == prev:
				continue
			for _ in range(100):
				results = s2s.dialog(src, prefix='dial', method='rand')
				logP, hyp = results[0]
				lines.append('\t'.join([src, hyp, '%.4f'%logP]))
			prev = src

		with open(fld + '/test_out.tsv', 'w', encoding='utf-8') as f:
			f.write('\n'.join(lines))


	elif 'summary' == mode:
		print(s2s.model.summary())

	else:
		raise ValueError



def get_model_fld(args):
	fld_data = DATA_PATH +'/' + args.data_name

	s2s_config = 'width%s_depth%s'%(
			(args.token_embed_dim, args.rnn_units, args.dropout),
			(args.encoder_depth, args.decoder_depth))
	if args.max_src_len != 90 or args.max_tgt_len != 30:
		s2s_config += '_len' + str((args.max_src_len, args.max_tgt_len))

	s2s_config = s2s_config.replace("'",'')
	fld_model = os.path.join(OUT_PATH, args.data_name + '_' + s2s_config)

	subfld = [args.name]
	if args.name == 'mtask':
		if args.interp:
			subfld.append('interp')
		if args.stddev > 0:
			subfld.append('std%.2f'%args.stddev)
		if args.wt_S2T > 0:
			subfld.append('ST%.2f'%args.wt_S2T)
		if args.wt_S2S > 0:
			subfld.append('SS%.2f'%args.wt_S2S)
		if args.wt_T2T > 0:
			subfld.append('TT%.2f'%args.wt_T2T)

	return fld_data, fld_model, '_'.join(subfld) + args.fld_suffix



if __name__ == '__main__':
	parser.add_argument('name')
	parser.add_argument('mode')
	parser.add_argument('--path_test', type=str)
	parser.add_argument('--skip', type=int, default=0)
	args = parser.parse_args()
	run_s2s(args.name, args.mode, args)




			