"""
AUTHOR: Xiang Gao (xiag@microsoft.com) at Microsoft Research
"""

from nltk.translate.bleu_score import sentence_bleu
from collections import defaultdict


def calc_nltk_bleu(ref, hyp, max_ngram=4, smoothing_function=None):
	return sentence_bleu(
		[ref.split()], 
		hyp.split(), 
		weights=[1./max_ngram]*max_ngram,
		)


def naacl_eval(path_hyp, path_ref, wt_len=0, len_only=False):
    # path_ref: each line is `src \t ref`
    # path_hyp: each line is `src \t hyp \t logP`

    path_out = path_hyp + '.eval_wt%.3f.tsv'%wt_len
        
    def read_file(path, full=False):

        d = defaultdict(list)
        for line in open(path, encoding='utf-8'):
            ss = line.strip('\n').split('\t')
            src = ss[0].strip()
            if full:
                d[src].append(ss[1:])
            else:
                d[src].append(ss[1])
        return d

    def rank_hyp(tuples, wt_len):
        # tuples is list of `(hyp, logP)`

        to_rank = []
        for hyp, logP in tuples:
            hyp = hyp.strip()
            score = float(logP) + wt_len * len(hyp.split())
            to_rank.append((score, hyp))
        hyps = []
        for  _, hyp in sorted(to_rank, reverse=True):
            if hyp not in hyps:		# only keep top distinct hyps
                hyps.append(hyp)
        return hyps

    d_ref = read_file(path_ref)
    d_hyp = read_file(path_hyp, full=True)

    ngrams = [1,2,3,4]
    sum_prec = dict([(ngram, 0) for ngram in ngrams])
    sum_recall = dict([(ngram, 0) for ngram in ngrams])
    n = 0
    sum_Nr = 0
    sum_len_hyp = 0
    sum_len_ref = 0

    header = ['src','Nr','len_hyp','len_ref']
    for ngram in ngrams:
        header += ['prec%i'%ngram,'recall%i'%ngram]
    lines = ['\t'.join(header)]

    for src in d_hyp:
        if src not in d_ref:
            continue

        n += 1
        refs = d_ref[src]
        hyps = rank_hyp(d_hyp[src], wt_len)
        Nr = min(len(refs), len(hyps))
        len_hyp = sum([len(hyp.split()) for hyp in hyps[:Nr]])/Nr
        len_ref = sum([len(ref.split()) for ref in refs[:Nr]])/Nr
        sum_Nr += Nr
        sum_len_hyp += len_hyp
        sum_len_ref += len_ref
        if len_only:
            continue

        if n % 10 == 0:
            print('evaluated %i src'%n)
        line = [src, '%i'%Nr, '%.2f'%len_hyp, '%.2f'%len_ref]
        for ngram in ngrams:
            mat = np.zeros((Nr, Nr))
            for i_hyp in range(Nr):
                for i_ref in range(Nr):
                    try: 
                        mat[i_hyp, i_ref] = calc_nltk_bleu(refs[i_ref], hyps[i_hyp], max_ngram=ngram)
                    except ZeroDivisionError:
                        pass
	    prec = np.mean(np.max(mat, axis=1))
	    recall = np.mean(np.max(mat, axis=0))
            sum_prec[ngram] += prec
            sum_recall[ngram] += recall
            line += ['%.4f'%prec, '%.4f'%recall]
        lines.append('\t'.join(line))

    print('sample#\t%i, avg ref# %.2f'%(n, sum_Nr/n))
    print('wt_len %.4f, avg hyp_len %.2f, avg_ref_len %.2f'%(wt_len, sum_len_hyp/n, sum_len_ref/n))
    if len_only:
        return
    for ngram in ngrams:
    	prec = sum_prec[ngram]/n
    	recall = sum_recall[ngram]/n
    	f1 = 2 * prec * recall / (prec + recall + 1e-9)
        print('%igram precision/recall/f1\t%.4f/%.4f/%.4f'%(
            ngram, prec, recall, f1))

    with open(path_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--path_hyp', type=str)
	parser.add_argument('--path_ref', type=str)
	parser.add_argument('--wt_len', type=float, default=0)
	parser.add_argument('-len_only', action='restore_true')
	args = parser.parse_args()
	naacl_eval(args.path_hyp, args.path_ref, wt_len=args.wt_len, len_only=args.len_only)
