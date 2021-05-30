# Implementing-Sparse-Hidden-Markov-Models
PiB 5 ECTS 2021

```python
import numpy as np
import pandas as pd
import seaborn as sns
import random 
import time
import math
```

```python
class hidden_markov:
    def __init__(self, init, trans, emis):
        self.init = init
        self.trans = trans
        self.emis = emis
```
        
 
```python       
def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]
```

```python
seq = "atatc"


init = np.array(
    [0.30, 0.10, 0.30, 0.30]
)

trans = np.array([
    [0.40, 0.10, 0.50, 0],
    [0.45, 0, 0.05, 0.5],
    [0, 0.20, 0.40, 0.2],
    [0, 0.3, 0.2, 0.5],
])

emis = np.array([  
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.1, 0.3, 0.2, 0.4],
])

hmm = hidden_markov(init,trans,emis)
```

```python
def dense2sparse(dense_array, transform=lambda x: x):
    # Number of rows in dense array
    N = len(dense_array)
    # Number of columns in dense array
    M = len(dense_array[0])


    sparse_array = []

    for i in range(0, N):
        for j in range(0, M):
            if dense_array[i, j] == 0:
                continue
            else:
                sparse_array.append((i, j, transform(dense_array[i, j])))
    return sparse_array

F = dense2sparse(trans)
print(F)
F = dense2sparse(trans, math.log)
print(F)
```


```python
def forward(X, hmm):
    if type(X) is str:
        X = translate_observations_to_indices(X)
    # Table size:
    N = len(X)
    K = len(hmm.init)
    F = np.zeros((K,N))


    # Initialize first column:
    for k in range(0, K):
        F[k, 0] = hmm.init[k] * hmm.emis[k, X[0]]

    # Recursive, the remaining of the table
    for i in range(1, N):
        for k in range(0, K):
            F[k,i] = 0  
            for kk in range(0,K):
                F[k,i] += F[kk,i-1] * hmm.trans[kk,k] * hmm.emis[k,X[i]]
                
    return F 


# print(forward(seq, hmm))
```

```python
def likelihood(X,hmm):
    F = forward(X,hmm)
    return sum(F[:,len(X)-1])

print(likelihood(seq,hmm))
```

```python
def scale_forward(X, hmm):
    if type(X) is str:
        X = translate_observations_to_indices(X)
    # Table size:
    N = len(X)
    K = len(hmm.init)
    F = np.zeros((K,N))
    scales = np.zeros(N)

    # Initialize first column:
    for k in range(0, K):
        F[k, 0] = hmm.init[k] * hmm.emis[k, X[0]]

    scales[0] = sum(F[:, 0])
    F[:,0] /= scales[0]

    # Recursive, the remaining of the table
    for i in range(1, N):
        for k in range(0, K):
            F[k,i] = 0
            for kk in range(0, K):
                F[k,i] += F[kk,i-1] * hmm.trans[kk,k] * hmm.emis[k,X[i]]
        scales[i] = sum(F[:,i])
        F[:,i] /= scales[i]

    return F, scales

# print(scale_forward(seq,hmm))
```

```python
def scale_log_likelihood(X, hmm):
    _, scales = scale_forward(X, hmm)
    return sum(np.log(scales))

print(scale_log_likelihood(seq, hmm))
```

```python
def sparse_scale_forward(X, hmm):
    if type(X) is str:
        X = translate_observations_to_indices(X)
    # Table size:
    N = len(X)
    K = len(hmm.init)
    F = np.zeros((K,N))
    scales = np.zeros(N)

    sparseT = dense2sparse(hmm.trans)

    # F[:] = 0

    # Initialize first column:
    for k in range(0, K):
        F[k, 0] = hmm.init[k] * hmm.emis[k, X[0]]

    scales[0] = sum(F[:, 0])
    F[:,0] /= scales[0]

    # Recursive, the remaining of the table
    for i in range(1, N):
        for l, k, t in sparseT:
            F[k, i] += F[l, i-1] * t * hmm.emis[k, X[i]]
        
        scales[i] = sum(F[:,i])
        F[:,i] /= scales[i]

    return F, scales

print(sparse_scale_forward(seq,hmm))
```


```python
def sparse_scale_log_likelihood(X, hmm):
    _, scales = sparse_scale_forward(X, hmm)
    return sum(np.log(scales))

print(sparse_scale_log_likelihood(seq, hmm))
```


```python
# Viterbi algorithm

def viterbi(X, hmm):
    X = translate_observations_to_indices(X)
    # Table size:
    N = len(X)
    K = len(hmm.init)

    # V table
    V = np.zeros((K,N))

    # Log transform all input probabilities
    IL = np.log(hmm.init)
    TL = np.log(hmm.trans)
    EL = np.log(hmm.emis)
    
    # Initalize, first column:
    for k in range(0, K): 
        V[k, 0] = IL[k] + EL[k, X[0]]
    
    # Rest of the table
    for i in range(1, N):
        for k in range(0, K):
            V[k, i] = -math.inf
            for kk in range(0, K):
                # V[k, i] largest as of yet
                V[k, i] = max(V[k,i], V[kk, i - 1] + TL[kk,k] + EL[k, X[i]])
                # V[k, i] largest after looking at kk
    return V

print(viterbi(seq,hmm))
```

```python
# Sparse Viterbi algorithm

def sparse_viterbi(X, hmm):
    X = translate_observations_to_indices(X)
    # Table size:
    N = len(X)
    K = len(hmm.init)

    # V table
    V = np.zeros((K,N))

    # Log transform all input probabilities
    IL = np.log(hmm.init)
    EL = np.log(hmm.emis)

    sparseT = dense2sparse(hmm.trans, math.log)
    
    V[:] = -math.inf
    

    # Initalize, first column
    for k in range(0, K):
        V[k, 0] = IL[k] + EL[k, X[0]]
    
    for i in range(1, N):
        for l, k, t in sparseT:
            V[k, i] = max(V[k, i], V[l, i - 1] + t + EL[k, X[i]])
        
    return V



print(sparse_viterbi(seq,hmm))
```


```python
# Making a transition matrix of random dimension
def random_trans(K, proportion):
    # Making a K x K array with random floats
    randomTM = np.random.rand(K, K)
    no_zeros = int(randomTM.size*proportion)

    # Adding zeros in random places
    index_zero = np.random.choice(randomTM.size, no_zeros, replace=False)
    randomTM.ravel()[index_zero] = 0
    

    # Scaling the rows - so they sum to one
    for i in range(0, K):
        scale = 0
        scale = sum(randomTM[i, :])
        if scale != 0:
            randomTM[i, :] /= scale


    return np.around(randomTM, 3)
```

```python
def random_emis(K):
    # Making an K x 4 array with random floats - one column for each observable state (ACGT)
    randomEM = np.random.rand(K, 4) 

    # Scaling the rows - so they sum to one
    for i in range(0, K):
        scale = 0
        scale = sum(randomEM[i, :])
        if scale != 0:
            randomEM[i, :] /= scale
    
    return np.around(randomEM, 2)
```

```python
# Making a sequence from 'random'
def random_seq(length):
    return ''.join(random.choice('ACTG') for _ in range(length))
```

```python
def time_algo_seqlength(ns, algo):
  ns_, times = [], []
  for n in ns:
    seq = random_seq(n)

    start = time.time()
    algo(seq, hmm)
    end = time.time()
    
    runningtime = end - start
    ns_.append(n)
    times.append(runningtime)
  return pd.DataFrame({'n': ns_, 'time': times})
```

```python
# Viterbi and Forward
# Reps and range

no_reps = 1 #Number of repetitions
min_n, max_n = 1, 1000 #Range of lenghts of n
ns = []
for n in range(min_n, max_n):
    ns.extend([n] * no_reps)

time_measure_forward = time_algo_seqlength(ns, forward, 0)
time_measures_viterbi = time_algo_seqlength(ns, viterbi, 0)

concatenated = pd.concat([time_measure_forward.assign(dataset='Forward algorithm'), time_measures_viterbi.assign(dataset='Viterbi algorithm')])
```

```python
g = sns.lmplot(x = 'n', y = 'time', 
                    hue = 'dataset', markers='.', scatter_kws={"s":40, "alpha":0.2}, palette = "Set1",
                    data = concatenated, fit_reg=False)
g.set(ylim = (0, max(concatenated['time'])))
```
```python
concatenated_vanilla['divided'] = concatenated_vanilla['time']/(concatenated_vanilla['n'])
g = sns.lmplot(x = 'n', y = 'divided', 
                    hue = 'dataset', markers='.', scatter_kws={"s":40, "alpha":0.15}, palette = "Set1",
                    data = concatenated_vanilla, fit_reg=False)
g.set(ylim = (0, 0.00005), xlabel = 'N', ylabel = 'Time [s]')
g._legend.set_title("Algorithm")
```
```python
def time_algo_matrix(ks, algo, proportion):
    ks_, times = [], []
    seq = random_seq(100)

    for k in ks:
      
        init = np.random.rand(k).round(2) 
        trans = random_trans(k, proportion)
        emis = random_emis(k)

        hmm = hidden_markov(init, trans, emis)
        
        start = time.time()
        algo(seq, hmm)
        end = time.time()
        
        runningtime = end - start
        ks_.append(k)
        times.append(runningtime)
        
    return pd.DataFrame({'k': ks_, 'time': times})

    print(time_algo_matrix(ks, scale_forward))
```

```python
no_reps = 1 #Number of repetitions
min_k, max_k = 1, 300 #Range of lenghts of k
ks = []
for k in range(min_k, max_k):
    ks.extend([k] * no_reps)

time_measure_viterbi = time_algo_matrix(ks, viterbi, 0)
time_measure_viterbi_90 = time_algo_matrix(ks, viterbi, 0.9)
time_measures_sparseviterbi = time_algo_matrix(ks, sparse_viterbi, 0)
time_measures_sparseviterbi_30 = time_algo_matrix(ks, sparse_viterbi, 0.3)
time_measures_sparseviterbi_60 = time_algo_matrix(ks, sparse_viterbi, 0.6)
time_measures_sparseviterbi_90 = time_algo_matrix(ks, sparse_viterbi, 0.9)
concatenated_viterbi = pd.concat(
    [time_measure_viterbi.assign(dataset='Viterbi (0%)'),
    time_measure_viterbi_90.assign(dataset='Viterbi (90%)'),
    time_measures_sparseviterbi.assign(dataset='Sparse Viterbi (0%)'),
    time_measures_sparseviterbi_30.assign(dataset='Sparse Viterbi (30%)'),
    time_measures_sparseviterbi_60.assign(dataset='Sparse Viterbi (60%)'),
    time_measures_sparseviterbi_90.assign(dataset='Sparse Viterbi (90%)')])
```

```python
g = sns.lmplot(x = 'k', y = 'time', 
                    hue = 'dataset', markers='.', scatter_kws={"s":40}, palette = "Set1",
                    data = concatenated_viterbi, fit_reg=False)
g.set(ylim = (0, 12), xlabel = 'K', ylabel = 'Time [s]')
g._legend.set_title("Algorithm and sparseness")
```


```python
concatenated_viterbi['divided'] = concatenated_viterbi['time']/(concatenated_viterbi['k']*concatenated_viterbi['k'])
g = sns.lmplot(x = 'k', y = 'divided', 
                    hue = 'dataset', markers='.', scatter_kws={"s":40}, palette = "Set1",
                    data = concatenated_viterbi, fit_reg=False)
g.set(ylim = (0, 0.0002), xlabel = 'K', ylabel = 'Time [s/K^2]')
g._legend.set_title("Algorithm and sparseness")
```








