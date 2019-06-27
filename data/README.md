The model requires `train.num`, `vali.num`, `test.num` and `vocab.txt`.
* `vocab.txt` is the vocab list of tokens. The first three token must be `_SOS_`, `_EOS_` and `_UNK_`, which represent "start of sentence", "end of sentence", and "unknown token".
* For these `*.num` files, the format for each line is `src \t tgt`, where `\t` is the tab delimiter, `src` is the context, and `tgt` is the response. 

The line ID (starts from 1) of `vocab.txt` is the token index used in `*.num` files. For examples, unknown tokens will be represented by `3` which is the token index of `_UNK_`. 
Both `src` and `tgt` are sequence of token index. 

You may build a vocab using the [build_vocab](https://github.com/golsun/NLP-tools/blob/master/data_prepare.py#L266) function to generate `vocab.txt`,
and then convert a raw text files to `*.num` 
(e.g. [train.txt](https://github.com/golsun/SpaceFusion/blob/master/data/toy/train.txt) to [train.num](https://github.com/golsun/SpaceFusion/blob/master/data/toy/train.num))
by the [text2num](https://github.com/golsun/NLP-tools/blob/master/data_prepare.py#L381) function

A [toy dataset](https://github.com/golsun/SpaceFusion/tree/master/data/toy) is provied as an example.
A [script](https://github.com/golsun/SpaceFusion/blob/master/data/switchboard.py) is provided to convert Switchboard to the formats we need. 