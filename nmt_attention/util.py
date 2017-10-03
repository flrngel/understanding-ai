import tensorflow as tf

class LookupTable(object):
  def __init__(self):
    self.dict = {}

  def lookup(self, word):
    if word not in self.dict:
      self.dict[word] = len(self.dict)
    return self.dict[word]

  def lookup_it(self, words):
    sequence = []
    for i in range(tf.size(words)):
      sequence.append(self.lookup(words[i]))
    return sequence
