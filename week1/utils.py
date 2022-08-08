from nltk import FreqDist
import nltk
import numpy as np
nltk.download('stopwords')
nltk.download('punkt')
sentence = ["Absolutely wonderful - silky and sexy and comfortable"]
# words = nltk.tokenize.word_tokenize(sentence)
# f = FreqDist(words)
# print(f.most_common())


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string

from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))
    
def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

def process_sentence(sentence):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    sentence = re.sub(r'\$\w*', '', sentence)
    sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence)
    sentence = re.sub(r'#', '', sentence)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    sentence_tokens = tokenizer.tokenize(sentence)

    sentence_clean = []
    for word in sentence_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            sentence_clean.append(stem_word)
    return sentence_clean
def build_word_freqs(tweets, process_func=process_tweet):
    """Build frequencies.
    Input:
        tweets: a list of tweets
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for tweet in tweets:
        for word in process_func(tweet):
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] = 1

    return freqs


def build_freqs(tweets, ys, process_func=process_tweet):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_func(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs
def extract_features(sentence, freqs):
    """
    It add the positive and negative frequencies of the words contatin in the sentence
    Input:
        sentence: a list of words for each sentence
        freqs: a dictionary coresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimention (1,3) for the sentence.
    """

    word_l = process_sentence(sentence)
    x = np.zeros((1,3))
    x[0,0] = 1 #    Bias term
    for word in word_l:
        x[0,1] += freqs.get((word, 1.0), 0)
        x[0,2] += freqs.get((word, 0.0), 0)

    assert (x.shape==(1,3))
    return x

def lookup(freqs, word, label):
    n = 0
    pair = (word, label)
    if pair in freqs:
        n = freqs[pair]

    return n

def count_tweet(results, tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in results:
                results[pair] += 1
            else:
                results[pair] = 1

    return results

def get_ratio(freqs, word):
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    pos_neg_ratio['positive'] = lookup(freqs,word,1)
    pos_neg_ratio['negative'] = lookup(freqs,word,0)
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1)/(pos_neg_ratio['negative'] + 1)
    return pos_neg_ratio

def get_words_by_threshold(freqs, label, threshold):
    word_list = {}
    for key in freqs.keys():
        word, _ = key
        pos_neg_ratio = get_ratio(freqs, word)
        if label == 1 and pos_neg_ratio['ratio'] >= threshold:
            word_list[word] = pos_neg_ratio
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:
            word_list[word] = pos_neg_ratio
    return word_list