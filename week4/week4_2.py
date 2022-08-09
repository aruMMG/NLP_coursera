import sys
sys.path.insert(0, './')
import pickle
import numpy as np

from week4.support_func import aproximate_knn, get_document_vecs, make_hash_table
from nltk.corpus import stopwords, twitter_samples
from nltk.tokenize import TweetTokenizer
en_embeddings_subset = pickle.load(open("week4/en_embeddings.p", "rb"))
fr_embeddings_subset = pickle.load(open("week4/fr_embeddings.p", "rb"))

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = all_positive_tweets + all_negative_tweets

document_vecs, ind2Tweet = get_document_vecs(all_tweets, en_embeddings_subset)
N_VECS = len(all_tweets)
N_DIMS = len(ind2Tweet[1])

N_PLANES = 10
N_UNIVERSES = 7    #   Number of time repeat the hashing to improve the search

np.random.seed(0)
planes_l = [np.random.normal(size=(N_DIMS, N_PLANES)) for _ in range(N_UNIVERSES)]


# np.random.seed(0)
# idx = 0
# planes = planes_l[idx]  # get one 'universe' of planes to test the function
# vec = np.random.rand(1, 300)
# print(f" The hash value for this vector,",
#       f"and the set of planes at index {idx},",
#       f"is {hash_value_of_vector(vec, planes)}")



# np.random.seed(0)
# planes = planes_l[0]  # get one 'universe' of planes to test the function
# vec = np.random.rand(1, 300)
# tmp_hash_table, tmp_id_table = make_hash_table(document_vecs, planes)

# print(f"The hash table at key 0 has {len(tmp_hash_table[0])} document vectors")
# print(f"The id table at key 0 has {len(tmp_id_table[0])}")
# print(f"The first 5 document indices stored at key 0 of are {tmp_id_table[0][0:5]}")

hash_tables = []
id_tables = []
for universe_id in range(N_UNIVERSES):  # there are 25 hashes
    print('working on hash universe #:', universe_id)
    planes = planes_l[universe_id]
    hash_table, id_table = make_hash_table(document_vecs, planes)
    hash_tables.append(hash_table)
    id_tables.append(id_table)

doc_id = 0
doc_to_search = all_tweets[doc_id]
vec_to_search = document_vecs[doc_id]

nearest_neighbor_ids = aproximate_knn(
    doc_id, vec_to_search, planes_l, hash_tables, id_tables, N_UNIVERSES, k=3, num_universes_to_use=5)

print(f"Nearest neighbors for document {doc_id}")
print(f"Document contents: {doc_to_search}")
print("")

for neighbor_id in nearest_neighbor_ids:
    print(f"Nearest neighbor at document id {neighbor_id}")
    print(f"document contents: {all_tweets[neighbor_id]}")