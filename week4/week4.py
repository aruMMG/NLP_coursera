import pickle
import string
import time
import sys
sys.path.insert(0, './')

from utils import process_tweet
from week4.support_func import cosine_similarity, get_dict, get_matrices

en_embeddings_subset = pickle.load(open("week4/en_embeddings.p", "rb"))
fr_embeddings_subset = pickle.load(open("week4/fr_embeddings.p", "rb"))

en_fr_train = get_dict('week4/en-fr.train.txt')
print('The length of the English to French training dictionary is', len(en_fr_train))
en_fr_test = get_dict('week4/en-fr.test.txt')
print('The length of the English to French test dictionary is', len(en_fr_train))

X_train, Y_train = get_matrices(
    en_fr_train, fr_embeddings_subset, en_embeddings_subset)