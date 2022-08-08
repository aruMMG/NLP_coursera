from distutils.command.build_scripts import first_line_re
from textblob import TextBlob
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from utils import build_freqs, take

# sentence = "Absolutely wonderful - silky and sexy and comfortable"
# print(TextBlob(sentence).sentiment)

# nltk.download('vader_lexicon')
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# sid = SentimentIntensityAnalyzer()
# print(sid.polarity_scores(sentence))


data_path = "/home/sakuni/side_work/sentiment_analysis/Customer_review.csv"

df = pd.read_csv(data_path)
print(df.head())
print(len(df))
clean_df = df.dropna(subset = ['ReviewText']).reset_index(drop=True)
print(len(clean_df))
# print(clean_df.loc[6295,:])
# print(clean_df.loc[6295,:])

# first, import the package (suppose we haven't imported it yet) run the analyzer (SentimentIntensityAnalyzer())
import nltk
import numpy as np
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Then create new columns for your dataframe (compound, pos, neu, neg) with empty dictionary
# With for loop, we get the scores for each review and append the scores to the dictionary

Result = { 'compound':[], 'pos':[] , 'neu':[], 'neg':[] }
count = 0
reviews = []
for review in clean_df['ReviewText']:
    count+=1
    reviews.append(review)
    score = sid.polarity_scores(review)
    Result['pos'].append(score['pos'])
    Result['neu'].append(score['neu'])
    Result['neg'].append(score['neg'])
    Result['compound'].append(score['compound'])

# Once this is done, new columns are created and dictionary is transformed to the dataframe
print(count)
clean_df['compound'] = pd.DataFrame(Result)['compound']
clean_df['pos'] = pd.DataFrame(Result)['pos']
clean_df['neu'] = pd.DataFrame(Result)['neu']
clean_df['neg'] = pd.DataFrame(Result)['neg']
clean_df['sentiment'] = np.where(clean_df['compound']>=0.05, 'Positive',np.where(clean_df['compound']<=-0.05,'Negative', 'Neutral'))
sorted_df = clean_df.sort_values(by=['compound'])
# print(sorted_df.head())
# print top 5 positive and negative
# print(len(sorted_df))
# print("Most positive #5 reviews ")
# print(sorted_df.tail())
# print("\n") # print line break
# print("Most negative #5 reviews ")
# print(sorted_df.head())

counts = sorted_df["sentiment"].value_counts()
fig1 = plt.figure()
plt.bar(counts.index, counts.values)
plt.xticks(counts.index, counts.index, fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.savefig("a.png")
plt.figure().clear()
f = build_freqs(reviews)

f = dict(sorted(f.items(), key=lambda item: item[1], reverse=True))
n_items = take(10, f.items())
print(n_items)
words = []
counts = []
for key in n_items:
    words.append(key[0])
    counts.append(key[1])
fig2 = plt.figure()
plt.barh(words,counts)
plt.title('Word Counts', fontsize=20, fontweight="bold")
# plt.ylabel('words', fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")

# plt.xlabel('counts', fontsize=14, fontweight="bold")
plt.savefig("b.png")
