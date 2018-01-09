import re

DIGITS = re.compile("[0-9]", re.UNICODE) # Normalize digits by replacing them with #

def get_word(sent, position, word_ids):
  if 0 <= position < len(sent):
    w = normalize_word(sent[position][0], word_ids)
  elif position == -1:
    w = '<S>'
  elif position == len(sent):
    w = '</S>'
  else:
    w = '<PAD>'
  return word_ids[w] if w in word_ids else -1, w # -1 denotes a missing item

def case_normalizer(word, dictionary):
  """
  In case the word is not available in the vocabulary, we can try multiple case normalizing procedure.
  We consider the best substitute to be the one with the lowest index, which is equivalent to the most frequent alternative.
  """
  lower = (dictionary.get(word.lower(), 1e12), word.lower())
  upper = (dictionary.get(word.upper(), 1e12), word.upper())
  title = (dictionary.get(word.title(), 1e12), word.title())
  results = sorted([(1e11, word), lower, upper, title])
  return results[0][1]

def normalize_word(word, word_ids):
  """ Find the closest alternative in case the word is OOV."""
  if not word in word_ids:
    word = DIGITS.sub("#", word)
  if not word in word_ids:
    word = case_normalizer(word, word_ids)
  return word if word in word_ids else '<UNK>'

def calc_coverage(word_ids, sents):
  word_types = {}
  covered_tokens, total_tokens, covered_sentences, total_sentences=(0,0,0,0)
  max_word_id=0 # we want to track it to see what is the number of the least frequent word in the dataset
  for s in sents:
    covered_tokens_in_sentence = 0
    for word in s: # word is a (token,pos) tuple
      total_tokens+=1
      word_types[word[0]] = 0
      normalized_word=normalize_word(word[0], word_ids)
      if normalized_word and normalized_word != '<UNK>':
        max_word_id = max(max_word_id, word_ids[normalized_word])
        covered_tokens_in_sentence +=1
        word_types[word[0]] = 1
    total_sentences += 1
    covered_sentences += 1 if covered_tokens_in_sentence == len(s) else 0
    covered_tokens += covered_tokens_in_sentence
  return [covered_sentences, total_sentences, covered_tokens, total_tokens, sum(word_types.values()), len(word_types), max_word_id, len(word_ids)]
