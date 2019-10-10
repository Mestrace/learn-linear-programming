# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'MSOR411\\Homework\5'))
    print(os.getcwd())
except:
    pass

# %%
import numpy as np
from numpy.linalg import inv, det

from itertools import combinations


def lp_init_bfs(c, G, h):
    m, n = G.shape
    G_idx = set(range(n))

    B_idxs = list(combinations(G_idx, m))

    N_idxs = []
    for B in B_idxs:
        N_idxs.append(tuple(G_idx - set(B)))

    optimal_idx = -1
    optimal_obj = float("inf")

    all_Xs = []

    for i, (B_idx, N_idx) in enumerate(zip(B_idxs, N_idxs)):
        i = i + 1

        B = G[:, B_idx]
        N = G[:, N_idx]
        print("Possible initial B matrices No.%d" % i)
        print("B index")
        print(B_idx)
        print("N index")
        print(N_idx)
        print("The value of B")
        print(B)
        print("The value of N")
        print(N)
        if det(B) == 0:
            print("B is a singular matrix; No bfs produced")
            continue
        
        print("The inverse of B")
        print(inv(B))
        Xb = inv(B) @ h
        Xn = np.zeros(n - m)
        print("The value of Xb")
        print(Xb)
        print("The value of Xn")
        print(Xn)

        X = np.hstack((Xb, Xn))
        X = X[np.argsort(np.hstack((B_idx, N_idx))), ]
        print("The solution")
        print(X)
        all_Xs.append(X)

        if np.any(X < 0):
            print("Solution does not satisfy non-negativity constraint; No bfs produced")
            continue
        else:
            cX = c @ X
            if cX < optimal_obj:
                optimal_obj = cX
                optimal_idx = i
            print("The objective is %3f" % float(cX))

        print()

    print()
    if optimal_idx != -1:
        print("Best initial B's is No.%d w/ objective value %2f" %
              (optimal_idx, optimal_obj))
    else:
        print("No initial B's has bfs; Check problem")

    return all_Xs[optimal_idx], optimal_obj


# %%
# Example 5.1
c = np.asarray([-3, 1, -1, 2])

G = np.asarray([
    [1, 0, 1, 2],
    [0, 1, 1, 2]
])
h = np.asarray([2, 3])

lp_init_bfs(c, G, h)


# %%
# Exercise 5.1.1
c = np.asarray([-1, 2, 1, -1])
G = np.asarray([
    [1, 3, 2, 0],
    [0, 1, 2, 1],
])
h = np.asarray([5, 6])
lp_init_bfs(c, G, h)


# %%
# Exercise 5.1.2
c = np.asarray([-2, -1, 1, 5, -3])
G = np.asarray([
    [3, 2, 1, 1, 0],
    [2, 2, 0, 3, 1],
])
h = np.asarray([3, 2])
lp_init_bfs(c, G, h)

# %%
c = np.asarray([1, -1, 2, 1, -1, 3, 1])
G = np.asarray([[1, 0, 2, 2, 0, 1, 0],
                [0, 1, 1, 3, 0, 4, 0], 
                [0, 0, 1, 2, 1, 4, 0], 
                [0, 0, 1, 3, 0, 1, 1]])
h = np.asarray([4,4,2,5])
lp_init_bfs(c, G, h)

# %%
# Exercise 5.3.3
c = np.asarray([2, -1, -4, 1, 1])
G = np.asarray(
    [[2, 3, 3, 1, 0],
     [3, 4, -1, 0, 1]]
)
h = np.asarray([23, 31])
lp_init_bfs(c, G, h)
#%%
