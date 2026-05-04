"""
cs5322f25prog3.py
Authors: Alicia Maranville, Fred Solis
Word sense disambiguation for words director, overtime, rubbish

Each function takes a list of sentences and returns list of integers in {1, 2} 
indicating the sense of the target word in each sentence. Uses pretrained models
loaded by pkl files.
"""

import os
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# preprocessing functions
def tokenize(text):
    """
    lowercase text and extract only letters/numbers (no punctuation)
    regex r"[a-z0-9]+" means find sequences of lowercase letters/digits
    """
    return re.findall(r"[a-z0-9]+", text.lower())

def context_window(tokens, target_word, n):
    """
    find target word in tokenized sentence
    return nearby words around it

    ex: ["the", "film", "director", "has", "a", "new", "movie"] with n = 2
    would find director at index 2 and return 2 words before and 2 words after it
    so, output word be ["the", "film" "has" "a"]
    """
    idx = -1
    for i, t in enumerate(tokens):
        if t == target_word or t.startswith(target_word): # starts with included in case of plural/related
            idx = i
            break
    lo = max(0, idx - n)
    hi = min(len(tokens), idx + n + 1)
    return tokens[lo:idx] + tokens[idx + 1:hi]


# ensemble class definition
class NBEnsemble:
    """Naive Bayes ensemble over multiple context window sizes."""

    def __init__(self, target_word, window_sizes=(2, 5, 10, 25)):
        self.target_word = target_word          # word to disambiguate
        self.window_sizes = list(window_sizes)  # different context sizes that will be used
        self.vectorizers = {}                   # stores a CountVectorizer per window size
        self.classifiers = {}                   # stores one NB classifier per window size

    def windowed_texts(self, sentences, n):
        """
        converts each sentence into clean strings by 
        1. tokenizing, 
        2. getting context window, 
        3. joining words back into a string

        returns list of preprocessed sentences
        """
        out = []
        for s in sentences:
            tokens = tokenize(s) # tokenize
            context = context_window(tokens, self.target_word, n) # get context
            out.append(" ".join(context)) # join words
        return out

    def fit(self, sentences, labels):
        """
        trains the ensemble by
        1. creates the context only text for each sentence
        2. vectorizes words into bag of words representation
        3. create naive bayes classifier: use multinomial because commonly used for NLP
        4. trains it on the sense labels
        5. saves vectorizer/classifier for the window size

        returns trained object
        """
        labels = np.array(labels)

        for n in self.window_sizes:

            texts = self.windowed_texts(sentences, n) # preprocess
            # convert to bag of words
            vec = CountVectorizer()
            X = vec.fit_transform(texts)

            clf = MultinomialNB() # classifier
            clf.fit(X, labels) # fit classifier

            # save vectorizer/classifier for this n
            self.vectorizers[n] = vec
            self.classifiers[n] = clf

        return self

    def predict(self, sentences):
        """
        predict the sense of the target word for each sentence provided
        builds context, uses trained vectorizer/classifier to get probabilities
        ensemble adds probabilities together, averages them, and chooses highest class

        returns list of predicted senses as integers
        """
        prob_sum = None
        classes = None

        for n in self.window_sizes:
            texts = self.windowed_texts(sentences, n) # context
            X = self.vectorizers[n].transform(texts) # vectorize
            probs = self.classifiers[n].predict_proba(X) # predict prob for each class
            if prob_sum is None:
                prob_sum = probs
                classes = self.classifiers[n].classes_
            else:
                prob_sum = prob_sum + probs

        avg = prob_sum / len(self.window_sizes) # average prob across window sizes
        idx = np.argmax(avg, axis=1) # get highest
        return [int(classes[i]) for i in idx] # return list

# word sense disambiguation functions
def WSD_Test_director(sentences):
    """
    disambiguate director
    - sense 1: organization leader
    - sense 2: film/play director
    """
    with open ("models/director_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model.predict(list(sentences))

def WSD_Test_overtime(sentences):
    """
    disambiguate overtime
    - sense 1: extra work hours
    - sense 2: extra play time in a tied game
    """
    with open ("models/overtime_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model.predict(list(sentences))

def WSD_Test_rubbish(sentences):
    """
    disambiguate rubbish
    - sense 1: trash/garbage
    - sense 2: nonsense/foolish talk)
    """
    with open ("models/rubbish_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model.predict(list(sentences))
