import pickle
import string
import time
import sys
sys.path.insert(0, './')

from utils import process_tweet
from week4.support_func import align_embeddings, cosine_similarity, get_dict, get_matrices, compute_cost, test_vocabulary

en_embeddings_subset = pickle.load(open("week4/en_embeddings.p", "rb"))
fr_embeddings_subset = pickle.load(open("week4/fr_embeddings.p", "rb"))

en_fr_train = get_dict('week4/en-fr.train.txt')
print('The length of the English to French training dictionary is', len(en_fr_train))
en_fr_test = get_dict('week4/en-fr.test.txt')
print('The length of the English to French test dictionary is', len(en_fr_train))

X_train, Y_train = get_matrices(
    en_fr_train, fr_embeddings_subset, en_embeddings_subset)
X_val, Y_val = get_matrices(en_fr_test, fr_embeddings_subset, en_embeddings_subset)
R_train = align_embeddings(X_train, Y_train, train_steps=400, learning_rate=0.8)
acc = test_vocabulary(X_val, Y_val, R_train)
print(f"accuracy on test set is {acc:.3f}")