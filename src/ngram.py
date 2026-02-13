import numpy as np
import string
import random

class UnigramModel:
  def __init__(self, text):
    # text is the training data, a string containing text from first 1000 examples
    self.text = text
    self.probs = {}

   # Build unigram probability distribution from training text
  def fit(self):
    for char in self.text:
      if char not in self.probs:
          self.probs[char] = 0
      self.probs[char] += 1
    
    for char in self.probs:
      self.probs[char] /= len(self.text)
  
  def predict(self, prefixes):
    # prefixes is a list of strings, each string is a prefix to predict the next character for
    # returns a list of predicted next characters, one for each prefix
    # Select the top three most probable characters to finish to prefix
    preds = []
    for _ in prefixes:
      sorted_chars = sorted(self.probs, key=self.probs.get, reverse=True)
      top_chars = sorted_chars[:3]
      preds.append(''.join(top_chars))
    return preds