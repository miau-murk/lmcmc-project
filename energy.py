import numpy as np
import scipy

import numpy as np
import scipy.stats
from scipy.special import logsumexp


def logp_func(x, loc1=-2.0, scale1=1.0, loc2=2.0, scale2=1.0, w=0.5, eps=1e-12):
    w = np.clip(w, eps, 1.0 - eps)
    logp1 = scipy.stats.norm.logpdf(x, loc=loc1, scale=scale1)
    logp2 = scipy.stats.norm.logpdf(x, loc=loc2, scale=scale2)
    a = np.log(w) + logp1
    b = np.log1p(-w) + logp2
    return logsumexp(np.stack([a, b], axis=0), axis=0)  # stable log(w*pdf1 + (1-w)*pdf2)

def dlogp_func(x, loc1=-2.0, scale1=1.0, loc2=2.0, scale2=1.0, w=0.5, eps=1e-12):
    w = np.clip(w, eps, 1.0 - eps)
    logp1 = scipy.stats.norm.logpdf(x, loc=loc1, scale=scale1)
    logp2 = scipy.stats.norm.logpdf(x, loc=loc2, scale=scale2)
    a = np.log(w) + logp1
    b = np.log1p(-w) + logp2
    logp = logsumexp(np.stack([a, b], axis=0), axis=0)
    r1 = np.exp(a - logp)          # in (0,1)
    r2 = 1.0 - r1
    dlogp1 = -(x - loc1) / (scale1 ** 2)
    dlogp2 = -(x - loc2) / (scale2 ** 2)
    return r1 * dlogp1 + r2 * dlogp2

def logp_dlogp_func(x, loc1=-2.0, scale1=1, loc2=2.0, scale2=1, w=0.75, eps=1e-12):
    return (logp_func(x, loc1, scale1, loc2, scale2, w, eps),
            dlogp_func(x, loc1, scale1, loc2, scale2, w, eps))



