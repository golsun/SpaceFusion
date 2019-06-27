"""
AUTHOR: Xiang Gao (xiag@microsoft.com) at Microsoft Research
convert Switchboard (Tiancheng Zhao's version, see following url) to txt files
https://github.com/snakeztc/NeuralDialog-CVAE/tree/master/data
"""

import pickle, json, sys, io, nltk


def cvae_tokenize(s):
	# follow Tiancheng's code
	return ' '.join(nltk.WordPunctTokenizer().tokenize(s.lower()))

def p2txt(path):
	data = pickle.load(open(path, 'rb'))
	for k in data:
		lines = []
		for i, d in enumerate(data[k]):
			if i%100 == 0:
				print('[%s]%i/%i'%(k, i, len(data[k])))
			txts = [cvae_tokenize(txt) for spk, txt, feat in d['utts']]
			for t in range(1, len(txts) - 1):
				src = ' EOS '.join(txts[:t])
				tgt = txts[t]
				lines.append(src + '\t' + tgt)
		with open(path + '_' + k + '.txt', 'w') as f:
			f.write('\n'.join(lines))

def json2txt(path):
	data = json.load(open(path))
	lines = []
	for i in range(len(data)):
		ctxts = []
		for ctxt in data[i]['context']:
			ctxt = ctxt.split(':')[1].strip()
			if ctxt == '<s> <d> </s>':
				continue
			else:
				ctxts.append(cvae_tokenize(ctxt))
		if len(ctxts) == 0:
			continue
		src = ' EOS '.join(ctxts)
		for tgt in data[i]['responses']:
			tgt = cvae_tokenize(tgt)
			lines.append(src+'\t'+tgt)

		print('dial %i/%i, got %i pairs'%(i, len(data), len(lines)))
	with open(path+'.txt', 'w', encoding='utf-8') as f:
		f.write('\n'.join(lines))


if __name__ == '__main__':
	path_orig = 'full_swda_clean_42da_sentiment_dialog_corpus.p'
	p2txt(path_orig)
	path_json = 'test_multi_ref.json'
	json2txt(path_json)
