import sys
sys.path.insert(0, './')
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from week3.support_func import cosine_similarity, euclidean, get_country, get_accuracy, compute_pca
from utils import get_vectors

data = pd.read_csv('week3/capitals.txt', delimiter=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']

# print first five elements in the DataFrame
# print(data.head(5))
word_embeddings = pickle.load(open("week3/word_embeddings_subset.p", "rb"))
print(len(word_embeddings))
print("dimension: {}".format(word_embeddings['Spain'].shape[0]))

king = word_embeddings['king']
queen = word_embeddings['queen']

print(cosine_similarity(king, queen))
print(get_country('Athens', 'Greece', 'Cairo', word_embeddings))

accuracy = get_accuracy(word_embeddings, data)
print(f"Accuracy is {accuracy:.2f}")

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town',
         'village', 'country', 'continent', 'petroleum', 'joyful']

X = get_vectors(word_embeddings, words)

print('You have 11 words each of 300 dimensions thus X.shape is:', X.shape)

result = compute_pca(X, 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

plt.savefig("week3/pcs_word_scatter.png")