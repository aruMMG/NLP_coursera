from asyncio import ThreadedChildWatcher
from matplotlib import pyplot as plt
import pandas as pd
import nltk
import os 
import matplotlib.pyplot as plt
from utils import build_freqs, process_sentence, extract_features
from model import gradientDecent, sigmoid
import numpy as np



data_path = "/home/sakuni/side_work/sentiment_analysis/archive/data.csv"
df = pd.read_csv(data_path)
# print(df.head())
# print(df.describe())
# print(df.isnull().sum()) #  Check for null value

g = df.groupby("Sentiment").agg(list)

# print(len(g.iloc[2,0]))
# print(len(g.loc["negative"]["Sentence"]))
positive_sent_list = g.loc["positive"]["Sentence"]
negative_sent_list = g.loc["negative"]["Sentence"]
neutral_sent_list = g.loc["neutral"]["Sentence"]
test_pos_index = int(len(positive_sent_list)*0.2)
test_neu_index = int(len(neutral_sent_list)*0.2)
test_neg_index = int(len(negative_sent_list)*0.2)

test_positive_list = positive_sent_list[:test_pos_index]
test_neutral_list = neutral_sent_list[:test_neu_index]
test_negative_list = negative_sent_list[:test_neg_index]
train_positive_list = positive_sent_list[test_pos_index:]
train_neutral_list = neutral_sent_list[test_neu_index:]
train_negative_list = negative_sent_list[test_neg_index:]
# print(len(train_negative_list))
# print(len(test_negative_list))

train_x = train_positive_list + train_negative_list
test_x = test_positive_list + test_negative_list

train_y = np.append(np.ones((len(train_positive_list), 1)), np.zeros((len(train_negative_list), 1)), axis=0)
test_y = np.append(np.ones((len(test_positive_list), 1)), np.zeros((len(test_negative_list), 1)), axis=0)

print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))

freqs = build_freqs(train_x, train_y, process_sentence)
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

print('This is an example of a positive sentence: \n', train_x[5])
print('\nThis is an example of the processed version of the tweet: \n', process_sentence(train_x[5]))



X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i,:] = extract_features(train_x[i], freqs)

Y = train_y
J, theta = gradientDecent(X, Y, np.zeros((3,1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

def predict_sentence(sentence, freqs, theta):
    x = extract_features(sentence, freqs)
    y_pred = sigmoid(np.dot(x,theta))

    return y_pred

for sentence in ['I am happy', 'I am bad', 'this should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print("{} -> {}".format(sentence, predict_sentence(sentence, freqs, theta)))

my_sentence = 'Its gaining :)'
p = predict_sentence(my_sentence, freqs, theta)
print(p)

def test_logistic_regression(test_x, test_y, freqs, theta):
    y_hat = []

    for sentence in test_x:
        y_pred = predict_sentence(sentence, freqs, theta)
        if y_pred>0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)
    
    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)
    return accuracy

tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

print('Label Predicted Tweet')
for x,y in zip(test_x,test_y):
    y_hat = predict_sentence(x, freqs, theta)

    if np.abs(y - (y_hat > 0.5)) > 0:
        print('THE TWEET IS:', x)
        print('THE PROCESSED TWEET IS:', process_sentence(x))
        print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_sentence(x)).encode('ascii', 'ignore')))