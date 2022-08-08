from matplotlib import pyplot as plt
import pandas as pd
from utils import build_freqs, process_sentence, extract_features
import numpy as np
import nltk
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples


all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


fig = plt.figure(figsize=(5, 5))
labels = 'Positives', 'Negative'
sizes = [len(all_positive_tweets), len(all_negative_tweets)] 
plt.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')
fig.savefig("images/week1_pie.png")
plt.figure().clear()

tweets = all_positive_tweets + all_negative_tweets
labels = np.append(np.ones((len(all_positive_tweets), 1)), np.zeros((len(all_negative_tweets), 1)), axis=0)

freqs = build_freqs(tweets, labels)

keys = ['happi', 'merri', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
        '❤', ':)', ':(', '😒', '😬', '😄', '😍', '♛',
        'song', 'idea', 'power', 'play', 'magnific']
data = []
for word in keys:
    pos = 0
    neg = 0
    if (word,1) in freqs:
        pos = freqs[(word,1)]
    if (word, 0) in freqs:
        neg = freqs[(word,0)]
    
    data.append([word, pos, neg])

fig1, ax = plt.subplots(figsize=(8,8))
x = np.log([x[1] + 1 for x in data])
y = np.log([x[2] + 1 for x in data])

ax.scatter(x,y)
plt.xlabel("Log Positive Count")
plt.ylabel("Log Negative Count")

for i in range(0, len(data)):
    ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)

ax.plot([0,9], [0,9], color = "red")
fig1.savefig("images/scater_word_freqs.png")
plt.figure().clear()


test_pos_index = int(len(all_positive_tweets)*0.2)
test_neg_index = int(len(all_negative_tweets)*0.2)

test_positive_list = all_positive_tweets[:test_pos_index]
test_negative_list = all_negative_tweets[:test_neg_index]
train_positive_list = all_positive_tweets[test_pos_index:]
train_negative_list = all_negative_tweets[test_neg_index:]
train_x = train_positive_list + train_negative_list
test_x = test_positive_list + test_negative_list

train_y = np.append(np.ones((len(train_positive_list), 1)), np.zeros((len(train_negative_list), 1)), axis=0)
test_y = np.append(np.ones((len(test_positive_list), 1)), np.zeros((len(test_negative_list), 1)), axis=0)

X = np.zeros((len(train_x), 3))
Y = np.zeros((len(train_x), 1))
for i in range(len(train_x)):
    X[i,:] = extract_features(train_x[i], freqs)
    Y[i] = 1 if X[i,2] > X[i,1] else 0
fig2, ax1 = plt.subplots(figsize = (8, 8))

colors = ['red', 'green']

# Color based on the sentiment Y
ax1.scatter(X[:,1], X[:,2], c=[colors[int(k)] for k in Y], s = 0.1)  # Plot a dot for each pair of words
plt.xlabel("Positive")
plt.ylabel("Negative")
fig2.savefig("images/scatter.png")