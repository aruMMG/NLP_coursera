from concurrent.futures import process
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, './')
from utils import process_tweet

def plot_vectors(vectors, colors=['k', 'b', 'r', 'm', 'c'], axes=None, fname='image.svg', ax=None):
    scale = 1
    scale_units = 'x'
    x_dir = []
    y_dir = []
    
    for i, vec in enumerate(vectors):
        x_dir.append(vec[0][0])
        y_dir.append(vec[0][1])
    
    if ax == None:
        fig, ax2 = plt.subplots()
    else:
        ax2 = ax
      
    if axes == None:
        x_axis = 2 + np.max(np.abs(x_dir))
        y_axis = 2 + np.max(np.abs(y_dir))
    else:
        x_axis = axes[0]
        y_axis = axes[1]
        
    ax2.axis([-x_axis, x_axis, -y_axis, y_axis])
        
    for i, vec in enumerate(vectors):
        ax2.arrow(0, 0, vec[0][0], vec[0][1], head_width=0.05 * x_axis, head_length=0.05 * y_axis, fc=colors[i], ec=colors[i])
    
    if ax == None:
        fig.savefig(fname)

def side_of_plane(P, v):
    sign_of_dotproduct = np.sign(np.dot(P, v.T))
    sign_of_dotproduct_scalar = sign_of_dotproduct.item()
    return sign_of_dotproduct_scalar

def hash_multi_plane(P_l, v):
    hash_value = 0
    for i, P in enumerate(P_l):
        sign = side_of_plane(P,v)
        hash_i = 1 if sign >=0 else 0
        hash_value += 2**i * hash_i
    return hash_value

def basic_hash_table(value_l, n_buckets):
    
    def hash_function(value, n_buckets):
        return int(value) % n_buckets
    
    hash_table = {i:[] for i in range(n_buckets)} # Initialize all the buckets in the hash table as empty lists

    for value in value_l:
        hash_value = hash_function(value,n_buckets) 
        hash_table[hash_value].append(value)
    
    return hash_table

def side_of_plane_matrix(P, v):
    dotproduct = np.dot(P, v.T)
    sign_of_dot_product = np.sign(dotproduct)
    return sign_of_dot_product

def hash_multi_plane_matrix(P, v, num_planes):
    side_matrix = side_of_plane_matrix(P,v)
    hash_value = 0
    for i in range(num_planes):
        sign = side_matrix.item(i)
        hash_i = 1 if sign>=0 else 0
        hash_value += 2**i * hash_i 
def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''

    dot = np.dot(A,B)
    norma = np.sqrt(np.dot(A,A))
    normb = np.sqrt(np.dot(B,B))
    cos = dot / (norma*normb)

    return cos

def get_dict(file_name):
    """
    This function returns the english to french dictionary given a file where the each column corresponds to a word.
    Check out the files this function takes in your workspace.
    """
    my_file = pd.read_csv(file_name, delimiter=' ')
    etof = {}  # the english to french dictionary to be returned
    for i in range(len(my_file)):
        # indexing into the rows.
        en = my_file.loc[i][0]
        fr = my_file.loc[i][1]
        etof[en] = fr

    return etof

def get_matrices(en_fr, french_vecs, english_vecs):
    """
    Input:
        en_fr: English to French dictionary
        french_vecs: French words to their corresponding word embeddings.
        english_vecs: English words to their corresponding word embeddings.
    Output: 
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the projection matrix that minimizes the F norm ||X R -Y||^2.
    """ 
    X_l = list()
    Y_l = list()

    english_set = english_vecs.keys()
    french_set = french_vecs.keys()

    french_words = set(en_fr.values())

    for en_word, fr_word in en_fr.items():
        if fr_word in french_set and en_word in english_set:
            en_vec = english_vecs[en_word]
            fr_vec = french_vecs[fr_word]
            X_l.append(en_vec)
            Y_l.append(fr_vec)

    X = np.vstack(X_l)
    Y = np.vstack(Y_l)

    return X, Y

def compute_cost(X, Y, R):
    m = X.shape[0]
    diff = np.dot(X,R)-Y
    diff_squared = diff**2
    sum_diff_squared = np.sum(diff_squared)
    loss = sum_diff_squared/m

    return loss

def compute_gradient(X,Y,R):
    m = X.shape[0]
    gradient = np.dot(X.transpose(), np.dot(X,R)-Y)*2/m

    return gradient

def align_embeddings(X, Y, train_steps=100, learning_rate=0.0003):
    np.random.seed(12)
    R = np.random.rand(X.shape[1], X.shape[1])

    for i in range(train_steps):
        if i%25 == 0:
            print(f"loss at iteration {i} is: {compute_cost(X,Y,R):.4f}")

        gradient = compute_gradient(X, Y, R)
        R -= learning_rate*gradient
    return R

def nearest_neighbor(v, candidates, k=1):
    similarity_l = []
    for row in candidates:
        cos_similarity = cosine_similarity(v, row)
        similarity_l.append(cos_similarity)

    sorted_ids = np.argsort(similarity_l)
    k_idx = sorted_ids[-k:]
    return k_idx
def test_vocabulary(X, Y, R):
    pred = np.dot(X,R)
    num_correct = 0

    for i in range(len(pred)):
        pred_idx = nearest_neighbor(pred[i], Y)
        if pred_idx == i:
            num_correct += 1
    accuracy = num_correct/len(pred)
    return accuracy

def get_document_embedding(tweet, en_embeddings):
    doc_embedding = np.zeros(300)
    process_doc = process_tweet(tweet)

    for word in process_doc:
        doc_embedding += en_embeddings.get(word, 0)
    return doc_embedding

def get_document_vecs(all_ducs, en_embeddings):
    ind2Doc_dict = {}
    documnet_vec_l = []

    for i, doc in enumerate(all_ducs):
        doc_emmbeding = get_document_embedding(doc, en_embeddings)
        ind2Doc_dict[i] = doc_emmbeding
        documnet_vec_l.append(doc_emmbeding)

    document_vec_matrix = np.vstack(documnet_vec_l)

    return document_vec_matrix, ind2Doc_dict

def hash_value_of_vector(v, planes):
    dot_product = np.dot(v, planes)
    sign_of_dot_product = np.sign(dot_product)
    h = sign_of_dot_product>=0
    h = np.squeeze(h)
    hash_value = 0

    n_planes = planes.shape[1]
    for i in range(n_planes):
        hash_value += np.power(2,i)*h[i]
    hash_value = int(hash_value)

    return hash_value

def make_hash_table(vecs,planes):
    num_of_planes = planes.shape[1]
    num_buckets = 2**num_of_planes
    hash_table = {i:[] for i in range(num_buckets)}
    id_table = {i:[] for i in range(num_buckets)}

    for i, v in enumerate(vecs):
        h = hash_value_of_vector(v, planes)
        hash_table[h].append(v)
        id_table[h].append(i)

    return hash_table, id_table

def aproximate_knn(doc_id, v, planes_l, hash_tables, id_tables, N_UNIVERSES, k=1, num_universes_to_use=False):
    if not num_universes_to_use:
        num_universes_to_use = N_UNIVERSES
    assert num_universes_to_use <= N_UNIVERSES
    vecs_to_consider_l = list()
    ids_to_consider_l = list()
    ids_to_consider_set = set()

    for universe_id in range(num_universes_to_use):
        planes = planes_l[universe_id]

        hash_value = hash_value_of_vector(v, planes)
        hash_table = hash_tables[universe_id]
        document_vec_l = hash_table[hash_value]
        id_table = id_tables[universe_id]
        new_id_to_consider = id_table[hash_value]

        if doc_id in new_id_to_consider:
            new_id_to_consider.remove(doc_id)

        for i , new_id in enumerate(new_id_to_consider):
            if new_id in ids_to_consider_set:
                document_vector_at_i = document_vec_l[i]

                vecs_to_consider_l.append(document_vector_at_i)
                ids_to_consider_l.append(new_id)
                ids_to_consider_set(new_id)

    vecs_to_consider_array = np.array(vecs_to_consider_l)
    nearest_neighbor_idx_l = nearest_neighbor(v, vecs_to_consider_array, k=k)
    nearest_neighbor_ids = [ids_to_consider_l[idx] for idx in nearest_neighbor_idx_l]

    return nearest_neighbor_ids

if __name__=="__main__":
    pass
    # np.random.seed(12)
    # m = 10
    # n = 5
    # X = np.random.rand(m,n)
    # Y = np.random.rand(m,n) * .1
    # R = align_embeddings(X, Y)
    # print(R)


    # v = np.array([1, 0, 1])
    # candidates = np.array([[1, 0, 5], [-2, 5, 3], [2, 0, 1], [6, -9, 5], [9, 9, 9]])
    # print(candidates[nearest_neighbor(v, candidates, 3)])
    
    # import pickle
    # en_embeddings_subset = pickle.load(open("week4/en_embeddings.p", "rb"))
    # custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
    # tweet_embedding = get_document_embedding(custom_tweet, en_embeddings_subset)
    # print(tweet_embedding[-5:])

