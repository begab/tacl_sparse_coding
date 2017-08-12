import re
import os
import sys
import pickle
import argparse
import numpy as np
import itertools

from sklearn.metrics import confusion_matrix

import spams
import pycrfsuite

from tacl_utils import get_word
from tacl_IO_helper import load_corpus, load_brown_clusters, load_embeddings, print_predictions, get_tag_dict

class SparseSeqTagger:
  templates = [(('w',i),) for i in range(-2,3)] # a more concise way of defining the feature set of https://github.com/chokkan/crfsuite/blob/master/example/pos.py
  templates += [(('w',i),('w',i+1)) for i in range(-2,2)] + [(('w',i),('w',i+1),('w',i+2)) for i in range(-2,1)] + [(('w',i),('w',i+1),('w',i+2),('w',i+3)) for i in range(-2,0)]
  templates += [(('w',-2), ('w',-1), ('w', 0), ('w', 1), ('w', 2))]
  templates += [(('w',0), ('w',i)) for i in range(-9,10) if i != 0]

  def __init__(self, params):
    self.crf_dir = '{}/'.format(params.crf_out_dir)
    if not os.path.exists(self.crf_dir):
      os.makedirs(self.crf_dir)
    self.K = params.K
    self.lda = params.lda
    if params.dense_vec_file is None and params.feature_mode == 'dense':
      print('For feature_mode>2 dense_vec_file parameter has to be set.')
      sys.exit(2)
    elif params.feature_mode == 'Brown':
      if params.brown_file is None:
        print('For feature_mode==2 brown_file parameter has to be set.')
        sys.exit(2)
      else:
         self.brown = load_brown_clusters(params.brown_file)
    self.dataset = params.dataset
    self.feature = params.feature_mode
    self.dict_constraint = params.dict_constraint
    self.alphas_nonneg = params.alphas_nonneg
    self.alphas_mode = '{}{}'.format(self.dict_constraint, 'NN' if self.alphas_nonneg else '')
    self.window_size = params.window_size
    self.use_word_form = params.use_word_form
    use_word_identity = 'WI' if self.use_word_form else 'noWI' # indicate whether word identity is used as a feature
    model_prefix = params.experiment_id if params.experiment_id else '{}{}'.format(params.lang, params.dataset)

    self.tags, self.tag_dict = get_tag_dict(self.dataset)
    self.lang = params.lang
    self.train_data = load_corpus(self.dataset, params.train_file, params.lang)
    self.test_data = load_corpus(self.dataset, params.test_file, params.lang)

    self.partial_training = params.partial_training
    if self.partial_training != -1:
      self.train_data = self.train_data[0:self.partial_training]

    self.words, self.embs, self.word_ids, self.id_words = load_embeddings(params.dense_vec_file)
    self.alphas = self.calc_alphas('{}/alphas'.format(os.path.dirname(params.dense_vec_file)))

    self.model_path='{}{}.crf'.format(self.crf_dir, '_'.join(map(str, [model_prefix, use_word_identity, self.window_size, self.feature])))
    if self.feature.startswith('FR'):
      self.model_path='{}{}.crf'.format(self.crf_dir, '_'.join(map(str, [model_prefix, self.window_size, self.feature])))
    elif self.feature.startswith('SC'):
      self.model_path='{}{}.crf'.format(self.crf_dir, '_'.join(map(str, [model_prefix, self.lda, self.K, use_word_identity, self.window_size, self.alphas_mode])))

  def calc_alphas(self, output_dir):
    alphas_file = '{}/{}-{}-{}.alph'.format(output_dir, self.lang, self.K, self.lda)
    if os.path.isfile(alphas_file) or 'False' in self.alphas_mode: # the unconstrained dictionary model needs to be pre-existent
      return pickle.load(open(alphas_file, 'rb'))

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
 
    param = {'K': self.K, 'lambda1': self.lda, 'numThreads': 4, 'batchsize': 400, 'iter': 1000, 'verbose': False}
    l_param ={x:param[x] for x in ['lambda1','pos','numThreads','verbose'] if x in param}
    l_param['return_reg_path'] = False
    l_param['pos'] = self.alphas_nonneg
    alphas = spams.lasso(self.embs.T, D=spams.trainDL(np.asfortranarray(self.embs.T), **param), **l_param).T
    pickle.dump(alphas, open(alphas_file, 'wb'), 2)
    return alphas

  @staticmethod
  def char_level_features(w):
    features = ['num={}'.format(w.isdigit()), 'cap={}'.format(w.istitle()),'sym={}'.format(all(not c.isalnum() for c in w))]
    for i in range(1,5):
      if len(w) >= i: # prefix and suffix features
        features.append('p{}={}'.format(i, w[:i]))
        features.append('s{}={}'.format(i, w[-i:]))
    return features

  def retrieve_standard_features(self, sent, t):
    if self.feature == 'FRwc':
      features = SparseSeqTagger.char_level_features(sent[t][0])
    else:
      features = []
    for template in SparseSeqTagger.templates:
      name = '|'.join(['%s[%d]' % (f, o) for f, o in template])
      values = []
      for field, offset in template:
        p = t + offset
        if p < 0 or p >= len(sent):
          values = []
          break
        values.append(sent[p][0])
      if values:
        features.append('%s=%s' % (name, '|'.join(values)))
    return features

  def word2featureshelper(self, sent, i, offset):
    feat_prefix = 'w{}_'.format(offset)
    position = i + offset
    wid, w = get_word(sent, position, self.word_ids)
    if self.use_word_form:
      features = {feat_prefix + (sent[position][0] if len(sent) > position >=0 else w): 1.}
    else:
      features = {}
    if self.feature == 'dense':
      features = dict(zip(map(lambda x: feat_prefix + 'dim_{}'.format(x), range(self.embs.shape[1])), self.embs[wid]))
    elif self.feature == 'Brown':
      cluster_id = self.brown.get(w)
      if cluster_id:
        for prefix_length in [4, 6, 10, 20]:
          features['brown_{}{}_{}'.format(feat_prefix, prefix_length, cluster_id[0:prefix_length])]=1.
    else:
      coeffs = self.alphas.getrow(wid).tocoo()
      for j, v in zip(coeffs.col, coeffs.data):
        features['{}{}dim_{}'.format(feat_prefix, 'P' if v > 0 else 'N', j)] = 1. # P for positive N for negative contribution
    return features

  def word2features(self, sent, i):
    if self.feature.startswith('FR'):
      features = self.retrieve_standard_features(sent, i)
    else:
      features = {'bias':1.}
      for offset in range(-self.window_size, self.window_size+1): # calculate features based on the self.windows_size many neighboring words
        features.update(self.word2featureshelper(sent, i, offset))
    return features

  def sent2features(self, sent):
    return [self.word2features(sent, token_index) for token_index in range(len(sent))]

  def sent2labels(self, sent):
    return [x[1] for x in sent]

  def train_model(self, reuse=True):
    if not(os.path.isfile(self.model_path)) or not reuse:
      X = [self.sent2features(sentence) for sentence in self.train_data]
      y = [self.sent2labels(sentence) for sentence in self.train_data]
      trainer = pycrfsuite.Trainer(verbose=False)
      for xseq, yseq in zip(X, y):
        trainer.append(xseq, yseq)
      # coefficient for L1 penalty, coefficient for L2 penalty, stop earlier, include transitions that are possible (but not observed)
      trainer.set_params({'c1': 1.0, 'c2': 1e-3,'max_iterations': 50, 'feature.possible_transitions': True})
      trainer.train(self.model_path)
    tagger = pycrfsuite.Tagger()
    tagger.open(self.model_path)
    if not reuse:
      os.remove(self.model_path)
    return tagger

  def run_experiment(self, tagger):
    preds, probs = [], []
    for x in [self.sent2features(sentence) for sentence in self.test_data]:
      prediction = tagger.tag(x)
      preds.append(prediction)
      probs.append(tagger.probability(prediction))

    y_test = [self.sent2labels(sentence) for sentence in self.test_data]
    golds = [test_labels for test_labels in y_test]
    perfect_sent_ids = [i for i in range(0, len(golds)) if golds[i]==preds[i]]

    golds = list(itertools.chain.from_iterable(golds))
    cm = confusion_matrix(golds, list(itertools.chain.from_iterable(preds)))
    info = tagger.info()
    if re.search('NER', self.model_path):
      out_file = self.model_path.replace('.crf', '{}.out'.format(self.partial_training))
      print_predictions(self.test_data, preds, out_file)
    accuracy = round(1.*np.diag(cm).sum()/cm.sum(), 4)
    sentence_level_accuracy = round(1.*len(perfect_sent_ids)/len(y_test), 4)
    return accuracy, sentence_level_accuracy

def main():
    parser = argparse.ArgumentParser(description="Run Sparse Sequence Labeling experiments")
    parser.add_argument("dataset", choices=['CONLLX', 'UNIV', 'NER'])
    parser.add_argument("--train_file", help="training file location", type=str, required=True)
    parser.add_argument("--test_file", help="test file location", type=str, required=True)
    parser.add_argument("--lang", help="language", type=str, required=True)
    parser.add_argument("--experiment_id", help="experiment ID to be included in the model file names", type=str, default=None)
    parser.add_argument("--dense_vec_file", help="location of the dense input vector to use", required=False)
    parser.add_argument("--brown_file", help="location of the files with the Brown clusters to use", required=False)

    dict_norm_constraing_parser = parser.add_mutually_exclusive_group(required=False)
    dict_norm_constraing_parser.add_argument('--dict_constraint_on', dest='dict_constraint', action='store_true')
    dict_norm_constraing_parser.add_argument('--dict_constraint_off', dest='dict_constraint', action='store_false')
    parser.set_defaults(dict_constraint=True)

    alphas_nonneg_parser = parser.add_mutually_exclusive_group(required=False)
    alphas_nonneg_parser.add_argument('--alphas_nonneg', dest='alphas_nonneg', action='store_true')
    alphas_nonneg_parser.add_argument('--alphas_any', dest='alphas_nonneg', action='store_false')
    parser.set_defaults(alphas_nonneg=False)

    parser.add_argument("--partial_training", help="number of the training sentences kept [default: -1]", type=int, default=-1)
    parser.add_argument("--lda", help="lambda for sparse coding [default: 0.1]", type=float, default=0.1)
    parser.add_argument("--K", help="# of basis vectors [default: 1024]", type=int, default=1024)
    parser.add_argument("--crf_out_dir", help="sets the model output directory [default:'./crf_models']", default='crf_models')
    parser.add_argument("--window_size", help="window size for feature generation [default: 1]", type=int, default=1)
    parser.add_argument("--feature_mode", help="feature mode [default: SC]", type=str, choices=['FRw', 'FRwc', 'Brown', 'dense', 'SC'], default='SC')
    
    wordform_option_parser = parser.add_mutually_exclusive_group(required=False)
    wordform_option_parser.add_argument('--with_word_form', dest='use_word_form', action='store_true')
    wordform_option_parser.add_argument('--without_word_form', dest='use_word_form', action='store_false')
    parser.set_defaults(use_word_form=False)

    args = parser.parse_args()
    #print(args)

    sst = SparseSeqTagger(args)
    tagger = sst.train_model(reuse=sst.partial_training == -1)
    acc, sentence_acc = sst.run_experiment(tagger)
    print('\t'.join(['Lang:', sst.lang, 'Model: ', sst.model_path, 'Per-token acc.:', str(acc), 'All correct sentences:',  str(sentence_acc)]))

if __name__ == "__main__":
  main()
