import sys
sys.path.insert(0, './')

import numpy as np
from week4.support_func import plot_vectors
import matplotlib.pyplot as plt



P = np.array([[1,2]])
perpenticular_plane = np.dot([[0,1], [-1,0]], P.T).T
fig, ax1 = plt.subplots(figsize = (8,8))

plot_vectors([P], colors=["b"], axes=[2,2], ax=ax1)
plot_vectors([perpenticular_plane*4, perpenticular_plane*-4], colors=["k", "k"], axes=[4,4], ax=ax1)

for i in range(20):
    v1 = np.array(np.random.uniform(-4,4,2))
    side_of_plane = np.sign(np.dot(P, v1.T))
    if side_of_plane == 1:
        ax1.plot([v1[0]], [v1[1]], 'bo')
    else:
        ax1.plot([v1[0]], [v1[1]], 'ro')
fig.savefig("week4/one_plane.png")