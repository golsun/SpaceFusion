
## Multi-ref Reddit
* Please first download `bz2` files from a third party, including the [comments](http://files.pushshift.io/reddit/comments/) and [submissions](http://files.pushshift.io/reddit/submissions/). For the SpaceFusion paper, we only used these from year 2011, but you may generate your own dataset with more/different years
* Then, please [this script](https://github.com/golsun/SpaceFusion/blob/master/data/reddit.py) to extract conversations from the raw bz2 files in the following steps for a given month `YYYY-MM`. E.g. for 2011 Jan
```
python data/reddit.py 2011-01 --task=extract --fld_bz2=[where/you/saved/bz2/files]
python data/reddit.py 2011-01 --task=conv --fld_bz2=[where/you/saved/bz2/files]
python data/reddit.py 2011-01 --task=ref --fld_bz2=[where/you/saved/bz2/files]
```
to generate the dataset used in the SpaceFusion paper, please use the default parameters.

## Switchboard
Switchboard data is downloaded from [this repo](https://github.com/snakeztc/NeuralDialog-CVAE/tree/master/data) and we provided a [script](https://github.com/golsun/SpaceFusion/blob/master/data/switchboard.py) to process that version to our format.

## Data format
The model requires `train.num`, `vali.num`, `test.num` and `vocab.txt`.
* `vocab.txt` is the vocab list of tokens. The first three token must be `_SOS_`, `_EOS_` and `_UNK_`, which represent "start of sentence", "end of sentence", and "unknown token".
* For these `*.num` files, the format for each line is `src \t tgt`, where `\t` is the tab delimiter, `src` is the context, and `tgt` is the response. 

The line ID (starts from 1, 0 is reserved for padding) of `vocab.txt` is the token index used in `*.num` files. For examples, unknown tokens will be represented by `3` which is the token index of `_UNK_`. 
Both `src` and `tgt` are sequence of token index. 

You may build a vocab using the [build_vocab](https://github.com/golsun/NLP-tools/blob/master/data_prepare.py#L266) function to generate `vocab.txt`,
and then convert a raw text files to `*.num` 
(e.g. [train.txt](https://github.com/golsun/SpaceFusion/blob/master/data/toy/train.txt) to [train.num](https://github.com/golsun/SpaceFusion/blob/master/data/toy/train.num))
by the [text2num](https://github.com/golsun/NLP-tools/blob/master/data_prepare.py#L381) function

A [toy dataset](https://github.com/golsun/SpaceFusion/tree/master/data/toy) is provied as an example following the format described above.
