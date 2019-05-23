from shared import *
from tf_lib import *
from main import run_s2s, get_model_fld
from scipy.optimize import fmin_powell as fmin
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import scipy

"""
AUTHOR: Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""



def euc_dist(a, b):
	# Euclidean distance
	if len(a.shape) == 1:
		return np.sqrt(np.sum(np.power(a - b, 2)))
	return np.sqrt(np.sum(np.power(a - b, 2), axis=1))

	

def dist_mat(coord):
	n = coord.shape[0]
	dist_T2T = np.zeros((n, n))
	for i in range(n):
		for j in range(i + 1, n):
			d = euc_dist(coord[i, :], coord[j, :])
			dist_T2T[i, j] = d
			dist_T2T[j, i] = d
	return dist_T2T




def interp(s2s, model_name, fld_save):

	n = 1000

	print('building data...')
	encoder_input_data, decoder_input_data, decoder_target_data, _, _ = s2s.dataset.load_data('test', max_n=n, check_src=True)
	vec_src = s2s.model_encoder['dial'].predict(encoder_input_data['dial'])
	vec_tgt = s2s.model_encoder['auto'].predict(encoder_input_data['auto'])

	print('evaluating...')
	uu = np.linspace(0, 1, 11)
	NLL = []
	for u in uu:
		latent = vec_src + u * np.ones(vec_src.shape) * (vec_tgt - vec_src)
		NLL_ = s2s.model_decoder_tf.evaluate(
				[latent, decoder_input_data], 
				decoder_target_data,
				verbose=0)
		print('u = %.3f, NLL = %.3f'%(u, NLL_))
		NLL.append(NLL_)

	
	fig = plt.figure(figsize=(6,3))
	ax = fig.add_subplot(111)
	ax.plot(uu, NLL,'k.-')
	print(uu)
	print(NLL)
	ax.plot(0, NLL[0], 'ro')
	ax.plot(1, NLL[-1], 'bo')

	ax.text(0, NLL[0] + 0.5, '  '+r'$S$', color='r')
	ax.text(1, NLL[-1], '  '+r'$T$', color='b')

	plt.xlabel(r'$u$')
	plt.ylabel('NLL')
	plt.title(model_name+'\nNLL of interpolation: '+r'$S+u(T-S)$')
	plt.subplots_adjust(top=0.8)
	plt.subplots_adjust(bottom=0.2)
	plt.savefig(fld_save+'/interp.png')

	with open(fld_save+'/interp.tsv','w') as f:
		f.write('\t'.join(['u'] + ['%.3f'%u for u in uu])+'\n')
		f.write('\t'.join(['NLL'] + ['%.3f'%l for l in NLL])+'\n')

	plt.show()






def clusters(s2s, model_name, fld_save, D=2):	
	# 2D visualization of the clouds of points /latent space

	n = 2000
	method = 'MDS'
	#method = 'tSNE'
	#method = 'isomap'

	AUTO = 'auto' in s2s.model_encoder

	print('building data...')
	encoder_input_data, _, _, source_texts, target_texts = s2s.dataset.load_data('test', max_n=n, check_src=False)

	extra_dial = []
	extra_auto = []
	extra_dial_txts = []
	extra_auto_txts = []

	for S_name in []:# ['trade','love']:
		with open('lists/%s.txt'%S_name) as f:
			lines = f.readlines()
		src_txt = lines[0].strip('\n')
		tgt_txts = [line.strip('\n') for line in lines[1:]]

		temp = np.zeros((1, s2s.dataset.max_seq_len))
		for t, token_index in enumerate(s2s.dataset.text2seq(src_txt)):
			temp[0, t] = token_index
		extra_dial.append(temp)
		extra_dial_txts.append(src_txt)

		for txt in tgt_txts:
			temp = np.zeros((1, s2s.dataset.max_seq_len))
			for t, token_index in enumerate(s2s.dataset.text2seq(txt)):
				temp[0, t] = token_index
			extra_auto.append(temp)
			extra_auto_txts.append(txt)

	data_dial = np.concatenate(extra_dial + [encoder_input_data['dial']], axis=0)
	latent_src = s2s.model_encoder['dial'].predict(data_dial)
	n_dial = data_dial.shape[0]
	if AUTO:
		data_auto = np.concatenate(extra_auto + [encoder_input_data['auto']], axis=0)
		latent_tgt = s2s.model_encoder['auto'].predict(data_auto)
		latent = np.concatenate([latent_src, latent_tgt], axis=0)
	else:
		latent = latent_src
	print('latent:', latent.shape)


	k = np.sqrt(latent_src.shape[1])
	f, ax = plt.subplots()
	cax = ax.imshow(dist_mat(latent)/k, cmap='bwr')
	ax.set_title(model_name)
	f.colorbar(cax)
	plt.savefig(fld_save+'/dist_mat.png')
	plt.close()

	def find_de(vec):
		mat = dist_mat(vec)/k
		de = []
		for i in range(vec.shape[0]):
			de.append(mat[i, i + 1:])
		return np.concatenate(de)

	f, axs = plt.subplots(2,1,sharex=True)
	de = find_de(latent_src)
	avg_de = np.mean(de)
	axs[0].hist(de, bins=50, color='r', alpha=0.5)
	axs[0].axvline(x=avg_de, color='r')
	title = model_name + '\nd(S,S) = %.2f'%avg_de

	if AUTO:
		de = find_de(latent_tgt)
		avg_de = np.mean(de)
		axs[0].hist(de, bins=50, color='b', alpha=0.5)
		axs[0].axvline(x=avg_de, color='b')
		title += ', d(T,T) = %.2f'%avg_de

		diff = latent_tgt - latent_src
		de = np.sqrt(np.mean(np.power(diff, 2), axis=1))
		avg_de = np.mean(de)
		axs[1].hist(de, bins=50, color='k', alpha=0.5)
		axs[1].axvline(x=avg_de, color='k')
		axs[1].set_title('d(S,T) = %.2f'%avg_de)

	axs[0].set_title(title)
	axs[-1].set_xlabel('Euc_dist/sqrt(%i)'%latent.shape[1])
	plt.savefig(fld_save+'/dist_hist.png')
	plt.close()
	#return

	if method == 'tSNE':
		approx = manifold.TSNE(init='pca', verbose=1).fit_transform(latent)
	elif method == 'MDS':
		approx = manifold.MDS(D, verbose=1, max_iter=1000, n_init=1).fit_transform(latent)
	elif method == 'isomap':
		approx = manifold.Isomap().fit_transform(latent)

	f, ax = plt.subplots()
	ax.plot(approx[:n_dial,0], approx[:n_dial,1], 'r.', alpha=0.2)
	ax.plot(approx[n_dial:,0], approx[n_dial:,1], 'b.', alpha=0.2)

	for i in range(len(extra_dial_txts)):
		plt.plot(approx[i, 0], approx[i, 1], 'ko', fillstyle='none')
		plt.text(approx[i, 0], approx[i, 1], extra_dial_txts[i], color='k')

	for i in range(len(extra_auto_txts)):
		plt.plot(approx[n_dial + i, 0], approx[n_dial + i, 1], 'ko', fillstyle='none')
		plt.text(approx[n_dial + i, 0], approx[n_dial + i, 1], extra_auto_txts[i], color='k')

	plt.title(model_name)
	plt.savefig(fld_save+'/%s_%s.png'%(method, int2str(n)))
	plt.title(' ')

	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()






def cos_sim(a, b):
	#return 1. - scipy.spatial.distance.cosine(a, b)
	return np.inner(a, b)/np.linalg.norm(a)/np.linalg.norm(b)


def angel_hist(s2s, model_name, fld_save):
	from rand_decode import load_1toN_data
	data = load_1toN_data(s2s.dataset.generator['test'])
	angel = []
	n = 1000

	extra_info = []

	for i in range(n):
		if i%10 == 0:
			print(i)
		d = data[i]
		src_seq = np.reshape(d['src_seq'], [1,-1])
		latent_src = np.ravel(s2s.model_encoder['dial'].predict(src_seq))
		diff = []
		for ref_seq in d['ref_seqs']:
			ref_seq = np.reshape(ref_seq, [1,-1])
			latent_ref = np.ravel(s2s.model_encoder['auto'].predict(ref_seq))
			diff.append(latent_ref - latent_src)

		for i in range(len(diff) - 1):
			for j in range(i + 1, len(diff)):
				if str(d['ref_seqs'][i]) == str(d['ref_seqs'][j]):
					continue
				angel.append(cos_sim(diff[i], diff[j]))
				extra_info.append('%i\t%i'%(i, len(d['ref_seqs'])))

	with open(fld_save+'/angel.txt', 'w') as f:
		f.write('\n'.join([str(a) for a in angel]))
	with open(fld_save+'/angel_extra.txt', 'w') as f:
		f.write('\n'.join(extra_info))

	plt.hist(angel, bins=30)
	plt.title(model_name)
	plt.savefig(fld_save+'/angel.png')
	plt.show()


def cos_sim(a, b):
	return np.inner(a, b)/np.linalg.norm(a)/np.linalg.norm(b)



if __name__ == '__main__':
	parser.add_argument('--name', default='mtask')
	parser.add_argument('--vis_tp', default='clusters')
	parser.add_argument('--S_name', default='trade')
	args = parser.parse_args()
	_, fld_model, model_name = get_model_fld(args)
	print('@'*20)
	print(model_name)
	print('@'*20)

	fld = os.path.join(fld_model, model_name, 'vis')
	print(fld)
	makedirs(fld)
	s2s = run_s2s(args.name, 'load', args)

	if args.vis_tp == 'interp':
		interp(s2s, model_name, fld)
	elif args.vis_tp == 'clusters':
		clusters(s2s, model_name, fld)
	elif args.vis_tp == 'angel':
		angel_hist(s2s, model_name, fld)
	else:
		raise ValueError