"""
train_models.py
Authors: Alicia Maranville and Fred Solis

Train multiple Multinomial Naive Bayes classifiers, 
each using a BOW representation of a different sized context window
around the target word. For testing, predictions are combined by
averaging the predicted probabilities.

Output: director_model.pkl, overtime_model.pkl, rubbish_model.pkl
"""

import os
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# importing the preprocessing/training functions from other file
from cs5322f25prog3 import NBEnsemble, tokenize as tokenize, context_window as context_window

def get_training_data(path, target_word):
    """
    process training file into (sentences, labels)
    file contains header (word, then two glasses)
    then sections titled 1 and 2 contained one sentence per line
    skipping blank lines and section headers

    returns list of sentences and list of labels 1 or 2
    """
    with open(path, "r", encoding="utf-8") as f: # open file
        lines = [ln.rstrip("\n") for ln in f] # remove newline characters

    sentences, labels = [], []
    current = None # sense 1/2 currently
    after_beginning = False # track to skip beginning of file with definitions

    for ln in lines:
        s = ln.strip()

        # skip blank lines
        if not s:
            continue

        # detect section headers 1 and 2
        if s == "1":
            current = 1
            after_beginning = True
            continue
        if s == "2":
            current = 2
            continue

        # skip before we see the first header
        if not after_beginning:
            continue

        # add sentences and current label to list
        sentences.append(s)
        labels.append(current)

    return sentences, labels

if __name__ == "__main__":
    words = ["director", "overtime", "rubbish"]
    summary = []

    for word in words:
        path = os.path.join("data/train", f"{word}.txt")
        sentences, labels = get_training_data(path, word) # gets training data from file
        n1 = sum(1 for l in labels if l == 1) # num of first sense
        n2 = sum(1 for l in labels if l == 2) # num of second sense

        # print info
        print(word)
        print(f"- {len(sentences)} sentences: {n1} for sense 1, {n2} for sense2")

        # train model on data and save
        alpha = 0.1 if word != "rubbish" else 0.5
        stopwords = True if word == "overtime" else False
        model = NBEnsemble(word, alpha=alpha, stopwords=stopwords).fit(sentences, labels)
        out = os.path.join("models", f"{word}_model.pkl")
        with open(out, "wb") as f:
            pickle.dump(model, f)
        print(f"- saved model to {out}")
