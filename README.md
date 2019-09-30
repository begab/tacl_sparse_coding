Accompanying code for the paper entitled 'Sparse Coding of Neural Word Embeddings for Multilingual Sequence'
==========

This repository contains the source code used in the [TACL paper](https://www.transacl.org/ojs/index.php/tacl/article/view/1063) entitled 'Sparse Coding of Neural Word Embeddings for Multilingual Sequence Labeling'.

## Online demo
The Part-of-Speech tagging models obtained by the code base can be [tested online for 29 languages](http://rgai.inf.u-szeged.hu/prosperAM/nlp).

## Sample code to try the code
Sample code for running the experiments for the English UDv1.2 treebank.
```bash
TRAIN_FILE=./ud-treebanks-v1.2/UD_English/en-ud-train.conllu
TEST_FILE=./ud-treebanks-v1.2/UD_English/en-ud-test.conllu
DENSE_EMBEDDING_FILE=./embs/polyglot/en.pkl
BROWN_FILE=./brown/en.brown.gz
for f in FRw FRwc Brown dense SC;
do
    python tacl_sequence_tagging.py UNIV --train_file $TRAIN_FILE --test_file $TEST_FILE --lang en --dense_vec_file $DENSE_EMBEDDING_FILE --brown_file=$BROWN_FILE --feature_mode $f;
done
```
Further sparse representations for various languages can be downloaded from [here](https://begab.github.io/sparse_embeds).
