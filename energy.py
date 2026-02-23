import numpy as np
import scipy
import scipy.stats
from scipy.special import logsumexp


################################################################################
# TEST FUNCTION 1
# 2D UNIMODUL NORMAL DISTRIBUTION
################################################################################

def logp_func_2d(x, loc=(0.0, 0.0), cov=((1.0, 0.0), (0.0, 1.0))):
    x = np.asarray(x)
    loc = np.asarray(loc)
    cov = np.asarray(cov)
    return scipy.stats.multivariate_normal.logpdf(x, mean=loc, cov=cov)  # [web:1]


def dlogp_func_2d(x, loc=(0.0, 0.0), cov=((1.0, 0.0), (0.0, 1.0))):
    x = np.asarray(x)
    loc = np.asarray(loc)
    cov = np.asarray(cov)

    dx = x - loc
    inv_cov = np.linalg.inv(cov)

    return -(dx @ inv_cov.T)  # [web:1]

def logp_dlogp_func_2d(x, loc=(0.0, 0.0), cov=((1.0, 0.0), (0.0, 1.0))):
    return logp_func_2d(x, loc=loc, cov=cov), dlogp_func_2d(x, loc=loc, cov=cov)  # [web:1]


################################################################################
# TEST FUNCTION 2
# 1D MULTIMODUL NORMAL DISTRIBUTION
################################################################################

def logp_func(x, loc1=0.0, scale1=1.0, loc2=0.0, scale2=1.0, w=0.5, eps=1e-12):
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

def logp_dlogp_func(x, loc1=0.0, scale1=1, loc2=0.0, scale2=1, w=0.5, eps=1e-12):
    return (logp_func(x, loc1, scale1, loc2, scale2, w, eps),
            dlogp_func(x, loc1, scale1, loc2, scale2, w, eps))


################################################################################
# TEST FUNCTION 3
# COS PERIODIC DISTRIBUTION
################################################################################

def wrap_pi(x):
    return (x + np.pi) % (2.0*np.pi) - np.pi

def logp_cos_torus(x, mu=0.0, beta=3.0, wrap_input=True):
    x = np.asarray(x)
    if wrap_input:
        x = wrap_pi(x)
    z = x - np.asarray(mu)
    return beta * np.cos(z)

def dlogp_cos_torus(x, mu=0.0, beta=3.0, wrap_input=True):
    x = np.asarray(x)
    if wrap_input:
        x = wrap_pi(x)
    z = x - np.asarray(mu)
    return -beta * np.sin(z)

def logp_dlogp_cos_torus(x, mu=0.0, beta=3.0, wrap_input=True):
    return logp_cos_torus(x, mu, beta, wrap_input), dlogp_cos_torus(x, mu, beta, wrap_input)


################################################################################
# TEST FUNCTION 4
# COMPLEX MULTIMODUL PERIODIC DISTRIBUTION (von Mises)
################################################################################

def _logsumexp(a, axis=None, keepdims=False):
    a = np.asarray(a)
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def logp_vm_mixture_torus(
    x,
    mus=None,        # (K,2) центры мод
    kappas=None,     # (K,2) концентрации по x/y (задают кривизну около моды)
    rhos=None,       # (K,) связь/скос: cos((x-mux)-(y-muy))
    logw=None,       # (K,) лог-веса смеси (не обязаны суммироваться в 1)
    wrap_input=True
):
    """
    Ненормированная logp на T^2: log sum_k w_k * exp( kx*cos(dx) + ky*cos(dy) + rho*cos(dx-dy) ).
    Периодичность 2π по каждой координате обеспечивается косинусами.
    """
    x = np.asarray(x, dtype=float)
    single = (x.ndim == 1)
    if wrap_input:
        x = wrap_pi(x)
    x = np.atleast_2d(x)  # (N,2)

    if mus is None:
        mus = np.array([[0.0, 0.0],[1.2, -1.0],[-1.5, 1.4]])
    if kappas is None:
        kappas = np.array([[5.0, 4.0], [5.0, 5.0], [2.0, 2.0]])
    if rhos is None:
        rhos = np.array([0.0, 0.0, 5.0])
    if logw is None:
        logw = np.log(np.array([0.5, 0.25, 0.25]))

    mus = np.asarray(mus, dtype=float)
    kappas = np.asarray(kappas, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    logw = np.asarray(logw, dtype=float)

    z = x[:, None, :] - mus[None, :, :]  # (N,K,2)
    if wrap_input:
        z = wrap_pi(z)

    zx, zy = z[..., 0], z[..., 1]
    kx, ky = kappas[None, :, 0], kappas[None, :, 1]
    rho = rhos[None, :]

    logc = logw[None, :] + kx*np.cos(zx) + ky*np.cos(zy) + rho*np.cos(zx - zy)  # (N,K)
    out = _logsumexp(logc, axis=1)  # (N,)
    return out[0] if single else out


def dlogp_vm_mixture_torus(x, mus=None, kappas=None, rhos=None, logw=None, wrap_input=True):
    """
    ∇ logp для смеси: responsibilities * ∇ log(component), где responsibilities = softmax(logc).
    """
    x = np.asarray(x, dtype=float)
    single = (x.ndim == 1)
    if wrap_input:
        x = wrap_pi(x)
    x = np.atleast_2d(x)  # (N,2)

    if mus is None:
        mus = np.array([[0.0, 0.0],[1.2, -1.0],[-1.5, 1.4]])
    if kappas is None:
        kappas = np.array([[5.0, 4.0], [5.0, 5.0], [2.0, 2.0]])
    if rhos is None:
        rhos = np.array([0.0, 0.0, 5.0])
    if logw is None:
        logw = np.log(np.array([0.5, 0.25, 0.25]))

    mus = np.asarray(mus, dtype=float)
    kappas = np.asarray(kappas, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    logw = np.asarray(logw, dtype=float)

    z = x[:, None, :] - mus[None, :, :]  # (N,K,2)
    if wrap_input:
        z = wrap_pi(z)

    zx, zy = z[..., 0], z[..., 1]
    kx, ky = kappas[None, :, 0], kappas[None, :, 1]
    rho = rhos[None, :]

    logc = logw[None, :] + kx*np.cos(zx) + ky*np.cos(zy) + rho*np.cos(zx - zy)  # (N,K)
    logp = _logsumexp(logc, axis=1, keepdims=True)  # (N,1)
    r = np.exp(logc - logp)  # (N,K)

    # ∂/∂x: -kx*sin(zx)  + d/dx [rho*cos(zx-zy)] = -rho*sin(zx-zy)
    # ∂/∂y: -ky*sin(zy)  + d/dy [rho*cos(zx-zy)] = +rho*sin(zx-zy)
    dlogc_dx = -kx*np.sin(zx) - rho*np.sin(zx - zy)
    dlogc_dy = -ky*np.sin(zy) + rho*np.sin(zx - zy)

    gx = np.sum(r * dlogc_dx, axis=1)
    gy = np.sum(r * dlogc_dy, axis=1)
    out = np.stack([gx, gy], axis=1)  # (N,2)
    return out[0] if single else out


def logp_dlogp_vm_mixture_torus(x, *args, **kwargs):
    return logp_vm_mixture_torus(x, *args, **kwargs), dlogp_vm_mixture_torus(x, *args, **kwargs)