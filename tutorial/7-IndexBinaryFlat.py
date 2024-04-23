

import numpy as np
import faiss

# Dimension of the vectors.
d = 256
nb = 1000000
nq = 10000
np.random.seed(1234)             # make reproducible
# Vectors to be indexed, each represented by d / 8 bytes in a nb
# i.e. the i-th vector is db[i].
db = np.random.randint(0, 256, size=(nb, d // 8), dtype=('uint8')) # ...initialize db...
#print(db)
# Vectors to be queried from the index.
queries = np.random.randint(0, 256, size=(nq, d // 8), dtype=('uint8')) # ...initialize queries...


# Initializing index.
index = faiss.IndexBinaryFlat(d)

# Adding the database vectors.
index.add(db)

# Number of nearest neighbors to retrieve per query vector.
k = 10

# Querying the index
D, I = index.search(queries, k)
print(I[:2])
print(D[:2])

# D[i, j] contains the distance from the i-th query vector to its j-th nearest neighbor.
# I[i, j] contains the id of the j-th nearest neighbor of the i-th query vector.