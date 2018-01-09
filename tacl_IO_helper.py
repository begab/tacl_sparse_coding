import os
import re
import sys
import gzip
import gensim
import codecs
import pickle
import numpy as np

from nltk.corpus.reader import DependencyCorpusReader
from universal_tags import convert
from itertools import chain, product

def get_tag_dict(mode):
  if mode == 'UNIV':
    tags = ['ADJ','ADP','ADV','AUX','CONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X']
  elif mode == 'CONLLX':
    tags = ['VERB','NOUN','PRON','ADJ','ADV','ADP','CONJ','DET','NUM','PRT','X','.']
  elif mode == 'NER':
    tags = ['-'.join(x) for x in product([c for c in 'BIES'], ['LOC', 'PER', 'ORG', 'MISC'])]+['O']
  elif 'MWE-' in mode:
    tags = ['-'.join(x) for x in product([c for c in 'BIE'], ['MWE_COMPOUND_ADJ', 'MWE_COMPOUND_NOUN', 'MWE_IDIOM', 'MWE_LVC', 'MWE_OTHER', 'MVE_VPC'])] + ['-'.join(x) for x in product([c for c in 'BIES'], ['NE_LOC', 'NE_PER', 'NE_ORG', 'NE_MISC'])] + ['O']
  return tags, dict(zip(tags, range(0, len(tags))))

def load_corpus(dataset, corpus_file, lang, transform_to_universal_pos_tags=True, train=None):
  corpus_id = None
  if dataset == 'UNIV':
    corpus = read_conllUD_file(corpus_file)
  elif dataset == 'CONLLX':
    univ_pos_mapper = {'bg':'btb', 'da':'ddt', 'nl':'alpino', 'pt':'bosque', 'sl':'sdt', 'es':'cast3lb', 'sv':'talbanken', 'tr':'tu-metusbanci', 'hu':'szeged', 'it':'isst', 'de':'tiger', 'en':'ptb'}
    if lang == 'en':
      corpus = read_conllX_file(corpus_file)
    else:
      corpus = read_conllX(corpus_file, '{}.conll'.format('train' if train else 'test'), lang)
    if transform_to_universal_pos_tags:
      if lang in univ_pos_mapper:
        univ_converter_id = univ_pos_mapper[lang] if lang == 'tr' else '{}-{}'.format(lang, univ_pos_mapper[lang])
      else:
        print('WARNING: no universal POS mapper file found for {}'.format(lang))
      corpus = [list(map(lambda x: (x[0], convert(univ_converter_id, x[1])), s)) for s in corpus]
  elif dataset == 'NER':
    corpus = extract_NER_sentences(lang, corpus_file)
  elif 'MWE-' in dataset:
    article_id = int(dataset.split('-')[1]) # this article is used for testing
    corpus = read_mwe(corpus_file, article_id, train)
  return corpus

def read_conllX(directory, suffix, lang):
  sentences = DependencyCorpusReader(directory, [fi for fi in os.listdir(directory) if fi.endswith(suffix)], encoding='utf8').tagged_sents()
  if lang == 'nl': # it is only necessary to split MWUs (multiword units) for the dutch corpus
    splitted_sentences = []
    for s in sentences:
      splitted_tokens = list(chain.from_iterable([t[0].split('_') for t in s]))
      splitted_pos = list(chain.from_iterable([t[1].split('_') for t in s]))
      splitted_sentences.append(zip(splitted_tokens, splitted_pos))
  else:
    splitted_sentences = sentences
  return enforce_unicode(splitted_sentences)

def read_conllX_file(f):
  sentences = []
  tokens = []
  for l in open(f, 'r'):
    s = l.split('\t')
    if len(s) == 14:
      tokens.append((s[1], s[4]))
    else:
      sentences.append(tokens)
      tokens = []
  return enforce_unicode(sentences)

def read_conllUD_file(location):
  sentences = []
  tokens = []
  with open(location, 'r') as f:
    for l in f:
      if not(l.strip().startswith('#')):
        s = l.split('\t')
        if len(s) == 10 and not('-' in s[0]):
          tokens.append((s[1], s[3]))
        elif len(l.strip())==0 and len(tokens) > 0:
          sentences.append(tokens)
          tokens = []
  return enforce_unicode(sentences)

def read_mwe(file_location, fold, train):
  '''
  Reads in the Wiki50 corpus.
  fold indicates the ID for the article which should be used for testing
  '''
  document = 0
  sentences=[]
  sentence = []
  with open(file_location) as f:
    for l in f:
      s=l.strip().split(' ')
      if s[0] == '-DOCSTART-':
        document+=1
      elif len(s) > 1 and s[0] != '-DOCSTART-' and ((train and document != fold) or (not train and document == fold)):
        sentence.append((s[0], 'O' if 'SENT_BOUND' in s[1] else s[1]))
      elif s[0] != '-DOCSTART-' and len(sentence) > 0:
        sentences.append(transform_labels_to_IOBES(sentence))
        sentence = []
  return enforce_unicode(sentences)

def extract_NER_sentences(lang, corpus_file):
  sentences=[]
  sentence = []
  with codecs.open(corpus_file, 'r', 'utf8' if lang in {'en'} else 'iso-8859-1') as f:
    for l in f:
      s=l.strip().split(' ')
      if len(s) > 1 and s[0] != '-DOCSTART-':
        sentence.append((s[0], s[-1]))
      elif s[0] != '-DOCSTART-' and len(sentence) > 0:
        sentences.append(transform_labels_to_IOBES(sentence))
        sentence = []
  return enforce_unicode(sentences)

def transform_labels_to_IOBES(sentence):
  prev_label = 'O'
  labels = [label for (token, label) in sentence]
  labels.append('O') # a dummy label to avoid indexing problems
  new_sentence = []
  for i in range(0, len(sentence)):
    l=labels[i]
    if l[0]=='I':
      if prev_label[0]=='O' and (labels[i+1][0]=='O' or labels[i+1][0]=='B'):
        l='S'+l[1:]
      elif prev_label[0]=='O':
        l='B'+l[1:]
      elif labels[i+1][0]=='O' or labels[i+1][0]=='B':
        l='E'+l[1:]
    elif l[0]=='B' and (labels[i+1][0]=='O' or labels[i+1][0]=='B'):
      l='S'+l[1:]
    new_sentence.append((sentence[i][0], l))
    prev_label=labels[i]
  return new_sentence

def print_predictions(test_sents, preds, outfile):
  """
  If the file already exists then it appends it on default.
  """
  output_dir = '/'.join(outfile.split('/')[0:-1])
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  with open(outfile, 'w') as f:
    for s in range(0,len(test_sents)):
      for t in range(0,len(test_sents[s])):
        f.write('{} {} {}\n'.format(test_sents[s][t][0], test_sents[s][t][1], preds[s][t]))
      f.write('\n\n')

def enforce_unicode(sentences):
  """
  In Python3 we should check for str class instead of unicode according to
  https://stackoverflow.com/questions/19877306/nameerror-global-name-unicode-is-not-defined-in-python-3
  """
  if len(sentences) == 0 or type(sentences[0][0][0]) == str: # if the first token is already unicode, there seems nothing to be done
    return sentences
  return [[(unicode(t[0], "utf8"), unicode(t[1], "utf8")) for t in s] for s in sentences]

def load_gzipped_embeddings(file_to_open):
  model = gensim.models.KeyedVectors.load_word2vec_format(file_to_open)
  temp_i2w = {}
  words = []
  nonzero_row_indices = []
  zero_embedding_counter = 0
  for v in model.vocab:
    vocab_entry = model.vocab[v]
    embedding = model.syn0[vocab_entry.index, :]
    if np.linalg.norm(embedding) > 0:
      words.append(vocab_entry)
      nonzero_row_indices.append(vocab_entry.index)
      temp_i2w[vocab_entry.index] = v
    else:
      zero_embedding_counter += 1
  i2w = {i:v[1] for i,v in enumerate(sorted(temp_i2w.items()))}
  w2i = {v:k for k,v in i2w.items()}
  embeddings = np.array(model.syn0[sorted(nonzero_row_indices)])
  return words, embeddings, w2i, i2w

def load_embeddings(dense_file):
  if dense_file.endswith('.gz'):
    return load_gzipped_embeddings(dense_file)
  try:
    f = open(dense_file, 'rb')
  except IOError as e:
    print('Error with opening the embedding file located at {}'.format(dense_file))
    sys.exit(1)
  try:
    words, embeddings = pickle.load(f)
  except:
    f.seek(0)
    words, embeddings = pickle.load(f, encoding='latin1')
  words, embeddings = filter_zero_rows(words, embeddings)
  return words, embeddings, {w:i for (i, w) in enumerate(words)}, dict(enumerate(words))

def filter_zero_rows(words, embeddings):
  np.seterr(all='raise')
  filtered_rows, filtered_words=[],[]
  unk_idx,unk_vec=-1,np.zeros(64)
  pad_idx,pad_vec=-1,np.zeros(64)
  s_idx,s_vec=-1,np.zeros(64)
  s_end_idx,s_end_vec=-1,np.zeros(64)
  for i in range(0, embeddings.shape[0]):
    r=embeddings[i]
    if words[i] == '<UNK>':
      unk_idx, unk_vec = i, r
    elif words[i] == '<PAD>':
      pad_idx, pad_vec = i, r
    elif words[i] == '<S>':
      s_idx, s_vec = i, r
    elif words[i] == '</S>':
      s_end_idx, s_end_vec = i, r
    if i<4 or np.any(r): # the first 4 symbols have special role, i.e. <s>, </s> <unk> and <pad>
      filtered_rows.append(r)
      filtered_words.append(words[i])
  filtered=np.asarray(filtered_rows)
  avg_vec = filtered.sum(axis=0) / (filtered.shape[0]-4)
  if not np.any(unk_vec):
    filtered[unk_idx]=avg_vec
  elif not np.any(pad_vec):
    filtered[pad_idx]=avg_vec
  elif not np.any(s_vec):
    filtered[s_idx]=avg_vec
  elif not np.any(s_end_vec):
    filtered[s_end_idx]=avg_vec
  return tuple(filtered_words), filtered

def load_brown_clusters(brown_file):
  brown={}
  if os.path.exists(brown_file):
    with gzip.open(brown_file, 'rt') as fin:
      for l in fin:
        path, word, _ = l.split("\t")
        brown[word] = path
  else:
    print('Brown file "{}" does not exist.'.format(brown_file))
  return brown
