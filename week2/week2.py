import sys
sys.path.insert(0, './')
from week1.utils import process_tweet, count_tweet, lookup, get_words_by_threshold
from naive_bayes_model_fun import train_naive_bayes, test_naive_bayes
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')

from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import string

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# avoid assumptions about the length of all_positive_tweets
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

freqs = count_tweet({}, train_x, train_y)

logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
print(logprior)
print(len(loglikelihood))

print("Naive Bayes accuracy = %0.4f" %
      (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))

words_positive = get_words_by_threshold(freqs, label=1, threshold=10)
print(words_positive)