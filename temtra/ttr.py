from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import temtra.littlemcmc.littlemcmc as lmc

Array = np.ndarray
LogpDlogpFunc = Callable[[Array], Tuple[Any, Array]]  # logp может быть float или ndarray (мы сведём к scalar)


@dataclass(frozen=True)
class HMCParams:
    step_scale: float = 0.25
    path_length: float = 4.0
    max_steps: int = 16
    target_accept: float = 0.8
    adapt_step_size: bool = False


def _ensure_starts_array(starts: Union[Array, Sequence[Array]], *, chains: int, ndim: int) -> Array:
    if isinstance(starts, np.ndarray):
        x = np.asarray(starts, dtype=float)
        if x.shape != (chains, ndim):
            raise ValueError(f"starts array must have shape (chains, ndim)={(chains, ndim)}, got {x.shape}")
        return x

    if len(starts) != chains:
        raise ValueError(f"starts list must have length={chains}, got {len(starts)}")

    xs = []
    for s in starts:
        s = np.asarray(s, dtype=float).reshape(-1)
        if s.size != ndim:
            raise ValueError(f"each start must have size ndim={ndim}, got shape={s.shape}")
        xs.append(s)
    return np.stack(xs, axis=0)


def _to_lmc_start_list(x: Array) -> List[Array]:
    return [np.asarray(x[i], dtype=float) for i in range(x.shape[0])]


def _sum_logp(logp: Any) -> float:
    return float(np.sum(np.asarray(logp)))


def _scaled_logp_dlogp(base_logp_dlogp: LogpDlogpFunc, beta: float) -> LogpDlogpFunc:
    beta = float(beta)

    def f(x: Array) -> Tuple[float, Array]:
        logp, dlogp = base_logp_dlogp(x)
        return beta * _sum_logp(logp), beta * np.asarray(dlogp, dtype=float)

    return f

def _get_quadpotential_diag(ndim: int):
    if hasattr(lmc, "quadpotential") and hasattr(lmc.quadpotential, "QuadPotentialDiag"):
        return lmc.quadpotential.QuadPotentialDiag(np.ones(ndim, dtype=float))
    if hasattr(lmc, "QuadPotentialDiag"):
        return lmc.QuadPotentialDiag(np.ones(ndim, dtype=float))
    raise AttributeError("Cannot find QuadPotentialDiag in littlemcmc")

def _extract_model_logp(stats: Dict[str, Any], *, key: str, chains: int) -> Array:
    v = np.asarray(stats[key])
    v = np.squeeze(v)
    if v.ndim == 0:
        v = v.reshape(1)
    if v.ndim == 1:
        pass
    elif v.ndim == 2:
        v = v[:, -1]
    else:
        raise ValueError(f"Unexpected stats['{key}'] shape after squeeze: {np.asarray(stats[key]).shape}")
    if v.shape[0] != chains:
        raise ValueError(f"stats['{key}'] has first dim {v.shape[0]}, expected chains={chains}")
    return v.astype(float, copy=False)


class HMCKernel:
    """HMC-ядро: один вызов littlemcmc.sample с заданными chains."""

    def __init__(
        self,
        *,
        logp_dlogp_func: LogpDlogpFunc,
        ndim: int,
        chains: int,
        params: HMCParams = HMCParams(),
        potential: Optional[Any] = None,
    ):
        self.ndim = int(ndim)
        self.chains = int(chains)
        self.logp_dlogp_func = logp_dlogp_func
        self.rng = np.random.default_rng()

        if potential is None:
            potential = _get_quadpotential_diag(self.ndim)

        self.step = lmc.hmc.HamiltonianMC(
            logp_dlogp_func=self.logp_dlogp_func,
            model_ndim=self.ndim,
            potential=potential,
            target_accept=float(params.target_accept),
            adapt_step_size=bool(params.adapt_step_size),
            step_scale=float(params.step_scale),
            path_length=float(params.path_length),
            max_steps=int(params.max_steps),
        )

    def one_draw(
        self,
        starts: Union[Array, Sequence[Array]],
        *,
        cores: int = 1,
    ) -> Tuple[Array, Dict[str, Any]]:
        x0 = _ensure_starts_array(starts, chains=self.chains, ndim=self.ndim)

        trace, stats = lmc.sample(
            logp_dlogp_func=self.logp_dlogp_func,
            model_ndim=self.ndim,
            step=self.step,
            draws=1,
            tune=0,
            chains=self.chains,
            cores=int(cores),
            start=_to_lmc_start_list(x0),
            random_seed=[int(self.rng.integers(0, 2**31 - 1)) for _ in range(self.chains)],
            discard_tuned_samples=True,
            progressbar=False,
        )

        trace = np.asarray(trace, dtype=float)  # (chains, 1, ndim)
        x1 = trace[:, 0, :].copy()
        return x1, stats


class HMC:
    def __init__(
        self,
        *,
        logp_dlogp_func: LogpDlogpFunc,
        ndim: int,
        chains: int,
        params: HMCParams,
        potential: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.ndim = int(ndim)
        self.chains = int(chains)
        self.logp_dlogp_func = logp_dlogp_func
        self.rng = np.random.default_rng() if rng is None else rng

        if potential is None:
            potential = _get_quadpotential_diag(self.ndim)
        self.potential = potential
        self.params = params

        self.step = lmc.hmc.HamiltonianMC(
            logp_dlogp_func=self.logp_dlogp_func,
            model_ndim=self.ndim,
            potential=self.potential,
            target_accept=float(params.target_accept),
            adapt_step_size=bool(params.adapt_step_size),
            step_scale=float(params.step_scale),
            path_length=float(params.path_length),
            max_steps=int(params.max_steps),
        )

    def draw(
        self,
        starts: Union[Array, Sequence[Array]],
        *,
        draws: int,
        tune: int = 0,
        cores: int = 1,
        progressbar: bool = True,
        discard_tuned_samples: bool = False,
    ) -> Tuple[Array, Dict[str, Any]]:
        x0 = _ensure_starts_array(starts, chains=self.chains, ndim=self.ndim)  # [file:63]
        trace, stats = lmc.sample(
            logp_dlogp_func=self.logp_dlogp_func,
            model_ndim=self.ndim,
            step=self.step,
            draws=int(draws),
            tune=int(tune),
            chains=self.chains,
            cores=int(cores),
            start=_to_lmc_start_list(x0),  # [file:63]
            random_seed=[int(self.rng.integers(0, 2**31 - 1)) for _ in range(self.chains)],  # [file:63]
            discard_tuned_samples=bool(discard_tuned_samples),
            progressbar=bool(progressbar),
        )
        return np.asarray(trace, dtype=float), stats

    @staticmethod
    def last_points(trace: Array) -> Array:
        # trace shape: (chains, draws_or_tune+draws, ndim)
        return np.asarray(trace[:, -1, :], dtype=float, copy=True)


class TemperedTransitions:

    def __init__(
        self,
        *,
        base_logp_dlogp: LogpDlogpFunc,
        betas: Sequence[float],
        ndim: int,
        chains: int,
        params: HMCParams = HMCParams(),
        potential: Optional[Any] = None,
        model_logp_key: str = "model_logp",
    ):
        self.base_logp_dlogp = base_logp_dlogp
        self.ndim = int(ndim)
        self.chains = int(chains)
        self.params = params
        self.potential = potential
        self.model_logp_key = str(model_logp_key)

        self.set_betas(betas)

    def set_betas(self, betas: Sequence[float]) -> None:
        betas = [float(b) for b in betas]
        if len(betas) < 2:
            raise ValueError("Tempered Transitions requires at least 2 betas.")
        if any(b <= 0.0 for b in betas):
            raise ValueError("All betas must be > 0 (we divide by beta to recover base logp).")

        self.betas = betas
        self.kernels: List[HMCKernel] = []
        for b in betas:
            lp = _scaled_logp_dlogp(self.base_logp_dlogp, b)
            self.kernels.append(
                HMCKernel(
                    logp_dlogp_func=lp,
                    ndim=self.ndim,
                    chains=self.chains,
                    params=self.params,
                    potential=self.potential,
                )
            )

    def _base_logp_from_stats(self, *, beta: float, stats: Dict[str, Any]) -> Array:
        if self.model_logp_key not in stats:
            raise ValueError(f"No '{self.model_logp_key}' in stats; cannot compute TT acceptance.")
        logp_beta = _extract_model_logp(stats, key=self.model_logp_key, chains=self.chains)
        return logp_beta / float(beta)

    def step(
        self,
        starts: Union[Array, Sequence[Array]],
        *,
        cores: int = 1,
        return_list: bool = False,
    ) -> Tuple[Union[Array, List[Array]], Dict[str, Any]]:
        x0 = _ensure_starts_array(starts, chains=self.chains, ndim=self.ndim)
        betas = self.betas
        L = len(betas)

        # base logp для k=0 (x_up[0]) берём прямым вызовом базовой функции
        base_logp_up: List[Array] = []
        logp0 = np.array([_sum_logp(self.base_logp_dlogp(x0[i])[0]) for i in range(self.chains)], dtype=float)
        base_logp_up.append(logp0)

        x_up: List[Array] = [x0]
        x = x0

        # Up pass: уровни 1..L-1
        for k in range(1, L):
            x, st = self.kernels[k].one_draw(x, cores=cores)
            x_up.append(x)
            base_logp_up.append(self._base_logp_from_stats(beta=betas[k], stats=st))

        # Down pass: уровни L-2..0
        x_down: List[Optional[Array]] = [None] * L
        base_logp_down: List[Optional[Array]] = [None] * L
        x_down[L - 1] = x_up[L - 1]
        base_logp_down[L - 1] = base_logp_up[L - 1]

        x = x_up[L - 1]
        for k in range(L - 2, -1, -1):
            x, st = self.kernels[k].one_draw(x, cores=cores)
            x_down[k] = x
            base_logp_down[k] = self._base_logp_from_stats(beta=betas[k], stats=st)

        x_prop = np.asarray(x_down[0], dtype=float)  # (chains, ndim)

        # log_alpha по цепям: sum_k db * (logp_down - logp_up)
        log_alpha = np.zeros(self.chains, dtype=float)
        for k in range(L - 1):
            db = betas[k] - betas[k + 1]
            log_alpha += db * (np.asarray(base_logp_down[k]) - np.asarray(base_logp_up[k]))

        log_u = np.log(np.random.default_rng().random(self.chains))
        accept = log_u < np.minimum(0.0, log_alpha)

        x_next = np.where(accept[:, None], x_prop, x0)

        info = {
            "accept": accept,            # shape (chains,)
            "log_alpha": log_alpha,      # shape (chains,)
            "betas": np.array(betas),
        }

        if return_list:
            return _to_lmc_start_list(x_next), info
        return x_next, info