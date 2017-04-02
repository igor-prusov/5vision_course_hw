
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

from multiprocessing import Pool

def mult_random_search(f, params_mean, params_std=1., n_workers=2, batch_size=100, n_iter=50):
    """ Multiprocessing version of Random Search algorithm."""
    params_std = np.ones_like(params_mean) * params_std
    best_params = params_mean
    ret_params = params_mean
    ys = np.zeros(batch_size)
    best_ys = 0
    pool = Pool(processes=n_workers)
    for _ in range(n_iter):
        # TO BE IMPLEMENTED: random search for parameters
        t = np.random.randn(batch_size, params_mean.size)
        r = np.linalg.norm(t - best_params, axis=0)
        d = t/r
        params  =  best_params + d
        # ys = np.array([f(p) for p in params])
        ys = np.array(pool.map(f, params))

        best_inds = ys.argsort()[::-1][0]
        best_params = params[best_inds]
        best_ys = ys[best_inds]


        # print np.max(ys), i
        yield {'results' : ys, 'best_params' : best_params}

def mult_cem(f, params_mean, params_std=1., n_workers=2, batch_size=100, n_iter=50, elite_frac=0.2):
    """ Multiprocessing version of CEM algorithm."""
    n_elite = int(np.round(batch_size * elite_frac))
    params_std = np.ones_like(params_mean) * params_std
    pool = Pool(processes=n_workers)
    for _ in range(n_iter):
        # TO BE IMPLEMENTED: CEM for search of parameters
        params = np.array([params_mean + dth for dth in params_std[None, :] * np.random.randn(batch_size, params_mean.size)])
        # ys = np.array([f(p) for p in params])
        ys = np.array(pool.map(f, params))
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_params = params[elite_inds]
        params_mean = elite_params.mean(axis=0)
        params_std = elite_params.std(axis=0)
        

        best_params = elite_params[0]

        yield {'results' : ys, 'best_params' : best_params}

