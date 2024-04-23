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

import faiss

nlist = 100
k = 6

index = faiss.index_factory(d, "IVF100,Flat") # Simplifying index construction

print(index.is_trained)

index.train(xb)
index.add(xb)

start_time = time.time()  # Record the start time

D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)

# start_time = time.time()  # Record the start time

# index.nprobe = 10              # make comparable with experiment above
# D, I = index.search(xq, k)     # search
# print(I[:1])
# print(D[:1])

end_time = time.time()  # Record the end time
print(f"Execution time: {end_time - start_time} seconds")
