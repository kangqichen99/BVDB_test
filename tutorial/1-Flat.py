# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import time

d = 64                           # dimension
nb = 1000000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# print(xb[:1])
# print(xq[:1])

import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)
#print(index)

k = 6                          # we want to see 4 nearest neighbors
# D, I = index.search(xb[:2], k) # sanity check
# print(I)
# print(D)

start_time = time.time()  # Record the start time

D, I = index.search(xq[:1], k)     # actual search
# print(I[:5])                   # neighbors of the 5 first queries
print(I[:1])                  # neighbors of the 5 last queries
print(D[:1])                  # neighbors of the 5 last queries

end_time = time.time()  # Record the end time
print(f"Execution time: {end_time - start_time} seconds")