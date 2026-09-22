"""Hierarchical resource profiling with uncertainty tracking.

Each model adapter (component) has intrinsic resource characteristics — workload
class, VRAM baseline, latency baseline, and scaling behavior with input size.
These characteristics are **learned online** from mixed (concurrent) observations
using Gaussian Process regression (Matérn-5/2 kernel) with interference-aware
heteroscedastic noise:

1. GP models resource usage as a continuous function of input_size.
2. Each observation carries its own noise variance R_i (solo: small, concurrent: large).
3. The GP posterior provides mean + variance at any input_size (including unseen).
4. Sliding window (N_max) bounds computation for long-running systems.

Hierarchy::

    Component (adapter-level)
    ├── workload_class: compute_bound | memory_bound | balanced
    ├── compute_profile: gpu_util EMA, arithmetic intensity
    └── ConfigProfile (per config_fingerprint)
        ├── gpu_baselines["*"]   →  ConfigBaseline (cross-GPU pooled, always present)
        └── gpu_baselines["0"]   →  ConfigBaseline (GPU-0 specific, optional)
            ├── vram_gp: GPEstimate (Matérn-5/2, continuous input_size)
            │   └── predict(x) → (μ*, σ²*) at any input_size
            ├── latency_gp: GPEstimate
            ├── vram_gp: GPEstimate (Matérn-5/2, sliding window)
            └── latency_gp: GPEstimate (Matérn-5/2, sliding window)

Lookup order for predictions:
    1. Exact gpu_id + exact input_size  →  most specific
    2. Exact gpu_id + scaling model interpolation  →  regression prediction
    3. Exact gpu_id + ``"*"`` input  →  cross-input fallback
    4. ``"*"`` gpu + same input_size lookup chain  →  cross-GPU fallback

This module is used by ``InterferenceRegistry`` and ``WorkloadClassifier`` to
provide richer priors for concurrent decomposition and interference prediction.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as _cf
import logging
import math
import multiprocessing as _mp
import os
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

_LOG = logging.getLogger(__name__)


_DRIFT_THRESHOLD: float = 0.30

_DRIFT_MIN_SAMPLES: int = 5



_SOLO_VARIANCE_EMA_ALPHA: float = 0.05

_INITIAL_CV_ASSUMPTION: float = 0.10


def _coerce_float(value: Any) -> float:
    """Preserve built-in float conversion while making the failure boundary explicit."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise


def _coerce_int(value: Any) -> int:
    """Preserve built-in int conversion while making the failure boundary explicit."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return _coerce_float(value)
    except Exception:
        return None


def _welford_update(
    n: int,
    mean: float,
    m2: float,
    value: float,
) -> tuple[int, float, float]:
    next_n = n + 1
    delta = value - mean
    next_mean = mean + delta / next_n
    next_m2 = m2 + delta * (value - next_mean)
    return next_n, next_mean, next_m2


@dataclass
class GaussianEstimate:
    """Online resource estimator using a scalar Kalman filter.

    Replaces the former Welford-based estimator.  The Kalman formulation
    allows each observation to carry its own measurement noise ``R_t``,
    so that high-noise concurrent observations are automatically down-
    weighted relative to clean solo observations.

    State-space model (scalar)::

        State:       μ_t = μ_{t-1} + w_t,   w_t ~ N(0, Q)
        Observation: y_t = μ_t + ε_t,        ε_t ~ N(0, R_t)

    Q (process noise) captures slow baseline drift (driver updates, thermal
    changes).  It is **not** related to interference — interference is handled
    entirely on the observation side via R_t.

    Initialisation uses a diffuse prior (P_0 → ∞), so the first observation
    sets the mean and P_1 = R_0 (De Jong 1991).

    Confidence is derived directly from the Kalman state covariance:
        confidence = max(0, 1 - P_t / P_1)
    """

    mean: float = 0.0
    n: int = 0
    _min: float = _coerce_float("inf")
    _max: float = _coerce_float("-inf")

    _P: float = _coerce_float("inf")
    _P1: float = _coerce_float(
        "inf"
    )
    _Q: float = 0.0

    _sigma2_solo: float = 0.0
    _sigma2_solo_initialised: bool = False

    _m2: float = 0.0


    @property
    def variance(self) -> float:
        """Population variance of *observations* (for std/cv display)."""
        return (self._m2 / self.n) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance) if self.variance > 0 else 0.0

    @property
    def sem(self) -> float:
        """Standard error of the mean (Kalman-derived)."""
        return (
            math.sqrt(self._P)
            if _coerce_float("inf") > self._P
            else _coerce_float("inf")
        )

    @property
    def ci_95(self) -> tuple[float, float]:
        """95% confidence interval for the mean."""
        margin = 1.96 * self.sem
        return (self.mean - margin, self.mean + margin)

    @property
    def cv(self) -> float:
        """Coefficient of variation (relative uncertainty)."""
        return (
            self.std / abs(self.mean) if abs(self.mean) > 1e-9 else _coerce_float("inf")
        )


    def update(self, x: float, *, R: float | None = None) -> None:
        """Incorporate a new observation via scalar Kalman update.

        Parameters
        ----------
        x : float
            The (possibly corrected) observation value.
        R : float, optional
            Measurement noise variance for this observation.
            - Solo observations: R = σ²_solo (small → high trust).
            - Corrected concurrent: R = σ²_solo/(1+δ)² + σ²_correction (medium).
            - Unknown concurrent: R = very large (low trust).
            If *None*, defaults to σ²_solo (treated as solo-quality).
        """
        self.n += 1

        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x
        old_mean_obs = self.mean if self.n > 1 else x

        if R is None:
            R = (
                self._sigma2_solo
                if self._sigma2_solo_initialised
                else ((_INITIAL_CV_ASSUMPTION * abs(x)) ** 2 if abs(x) > 1e-9 else 1.0)
            )

        if _coerce_float("inf") == self._P:
            self.mean = x
            self._P = R
            self._P1 = R
            if not self._sigma2_solo_initialised:
                self._sigma2_solo = (_INITIAL_CV_ASSUMPTION * abs(x)) ** 2
                self._sigma2_solo_initialised = True
            self._m2 = 0.0
            return

        P_pred = self._P + self._Q

        S = P_pred + R
        K = P_pred / S if S > 1e-30 else 0.0
        innovation = x - self.mean
        self.mean += K * innovation
        self._P = (1.0 - K) * P_pred

        delta = x - old_mean_obs
        delta2 = x - self.mean
        self._m2 += delta * delta2

    def update_solo_variance(self, innovation: float) -> None:
        """Update adaptive σ²_solo estimate from a solo observation's innovation.

        Called externally when the caller knows this was a solo observation.
        Uses EMA of squared innovations (Mehra 1970, adaptive variant).
        """
        sq = innovation * innovation
        if not self._sigma2_solo_initialised:
            self._sigma2_solo = sq
            self._sigma2_solo_initialised = True
        else:
            alpha = _SOLO_VARIANCE_EMA_ALPHA
            self._sigma2_solo = (1.0 - alpha) * self._sigma2_solo + alpha * sq

    def update_process_noise(self) -> None:
        """Adaptively estimate Q from innovation variance (Mehra 1970).

        Should be called periodically (e.g., every N solo observations).
        """
        if self._sigma2_solo_initialised and _coerce_float("inf") > self._P:
            pass


    def confidence(self) -> float:
        """Confidence score in [0, 1] derived from Kalman state covariance.

        confidence = max(0, 1 - P_t / P_1)

        P_1 is the covariance after the first observation (= R_0 under
        diffuse prior).  As more observations (especially solo) arrive,
        P_t shrinks and confidence rises.
        """
        if self.n == 0 or self._P1 <= 0 or _coerce_float("inf") == self._P1:
            return 0.0
        raw = 1.0 - self._P / self._P1
        return round(max(0.0, min(1.0, raw)), 3)

    def is_sufficient(self, required_relative_error: float) -> bool:
        """Check if the estimate meets the required relative precision.

        Uses the Kalman posterior 95% CI half-width relative to the mean:

            1.96 × √P_t / |μ̂_t| ≤ ε

        This is backed by the Cramér-Rao Lower Bound — P_t achieves the
        CRLB for Q=0 (Kay 1993) and the posterior CRLB for Q>0
        (Tichavský et al. 1998).  See <docs>
        .4 for full derivation and references [12]-[17].

        Parameters
        ----------
        required_relative_error : float
            Maximum acceptable relative error (e.g., 0.05 for 5%).

        Returns
        -------
        bool
            True if the 95% CI half-width is within ε of |mean|.
        """
        if self.n == 0 or _coerce_float("inf") <= self._P:
            return False
        if abs(self.mean) < 1e-9:
            return 1.96 * math.sqrt(self._P) < 1.0
        ci_half = 1.96 * math.sqrt(self._P)
        return ci_half <= required_relative_error * abs(self.mean)

    @property
    def relative_error(self) -> float:
        """Current relative estimation error (95% CI half-width / |mean|).

        Returns inf if mean is near zero or no observations.
        """
        if self.n == 0 or _coerce_float("inf") <= self._P or abs(self.mean) < 1e-9:
            return _coerce_float("inf")
        return 1.96 * math.sqrt(self._P) / abs(self.mean)


    def as_dict(self) -> dict[str, Any]:
        ci = self.ci_95
        re = self.relative_error
        return {
            "mean": round(self.mean, 3),
            "std": round(self.std, 3),
            "n": self.n,
            "ci_95_lower": round(ci[0], 3),
            "ci_95_upper": round(ci[1], 3),
            "confidence": self.confidence(),
            "relative_error": round(re, 4) if re < _coerce_float("inf") else None,
            "min": round(self._min, 3) if self._min != _coerce_float("inf") else None,
            "max": round(self._max, 3) if self._max != _coerce_float("-inf") else None,
        }

    @property
    def sigma2_solo(self) -> float:
        """Current estimate of solo measurement variance."""
        return self._sigma2_solo



_GP_N_MAX: int = 200

_GP_MAX_AGE: float = 7200.0

_GP_HP_OPT_MIN_N: int = 5

_GP_HP_OPT_INTERVAL: int = 20

_GP_JITTER: float = 1e-6

_GP_PROCESS_POOL: _cf.ProcessPoolExecutor | None = None
try:
    _GP_PROCESS_POOL_LOCK: threading.Lock = threading.Lock()
except RuntimeError as exc:
    raise RuntimeError("failed to initialize GP process-pool lock") from exc
_GP_PROCESS_POOL_WORKERS: int = 2


def _gp_process_start_method() -> str:
    """Return the process start method used for GP HP optimisation workers.

    Linux defaults to ``fork`` for ProcessPoolExecutor.  That is unsafe in the
    gateway because gRPC aio has C-core background threads; forking after those
    threads are active can produce ``fork_posix`` warnings and, in compressed
    mock runs, gateway segfaults.  ``spawn`` starts a fresh interpreter instead
    of inheriting the live gRPC runtime.
    """
    try:
        raw = str(os.environ["PROTON_GP_PROCESS_START_METHOD"] or "")
    except KeyError:
        raw = "spawn"
    method = raw.strip().lower()
    if method not in {"spawn", "forkserver", "fork"}:
        _LOG.warning(
            "[gp-process-pool] invalid PROTON_GP_PROCESS_START_METHOD=%r; using spawn",
            raw,
        )
        return "spawn"
    return method


def _gp_pool_ping(index: int = 0) -> int:
    """Picklable warm-up target for the GP process pool."""
    return index


def _get_gp_process_pool() -> _cf.ProcessPoolExecutor:
    """Return the module-level ProcessPoolExecutor, initialising it once."""
    global _GP_PROCESS_POOL
    if _GP_PROCESS_POOL is None:
        with _GP_PROCESS_POOL_LOCK:
            if _GP_PROCESS_POOL is None:
                ctx = _mp.get_context(_gp_process_start_method())
                _GP_PROCESS_POOL = _cf.ProcessPoolExecutor(
                    max_workers=_GP_PROCESS_POOL_WORKERS,
                    mp_context=ctx,
                )
    return _GP_PROCESS_POOL


def warm_gp_process_pool(timeout_s: float = 10.0) -> None:
    """Pre-start GP optimisation workers before gateway gRPC traffic begins.

    ProcessPoolExecutor launches workers lazily on first submit.  Warming it at
    gateway boot ensures later resource-signal updates do not need to create
    child processes while gRPC RPCs are active.
    """
    pool = _get_gp_process_pool()
    futures = [
        pool.submit(_gp_pool_ping, idx) for idx in range(_GP_PROCESS_POOL_WORKERS)
    ]
    for idx, fut in enumerate(futures):
        result = fut.result(timeout=timeout_s)
        if result != idx:
            raise RuntimeError(
                f"GP process pool warm-up returned unexpected result {result!r}"
            )


def _gp_hp_grid_search(
    xs: list[float],
    ys: list[float],
    rs: list[float],
    beta0: float,
    beta1: float,
    lengthscale: float,
    sigma2_f: float,
) -> tuple[float, float, int]:
    """Module-level picklable free function for ProcessPoolExecutor.

    Runs the 6×4 grid search over (lengthscale, sigma2_f) using log-marginal
    likelihood and returns (best_lengthscale, best_sigma2_f, n).  Must import
    numpy and math itself because worker processes do not inherit the main
    process's imported state.

    fix: extracted from ``GPEstimate._optimize_hyperparameters_from_snapshot``
    so it can be submitted to a ProcessPoolExecutor (bound methods are not
    picklable).
    """
    import math as _math

    import numpy as _np

    n = len(xs)
    if n < _GP_HP_OPT_MIN_N:
        return lengthscale, sigma2_f, n

    X = _np.array(xs)
    Y = _np.array(ys)
    R_diag = _np.array(rs)
    jitter = _GP_JITTER

    x_range = (_np.max(X) - _np.min(X)).item() if len(X) > 1 else 100.0
    y_var = _np.var(Y).item() if len(Y) > 1 else 1.0

    best_lml = -_math.inf
    best_l = lengthscale
    best_sf = sigma2_f

    sqrt5 = _math.sqrt(5.0)
    for l_frac in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        l_candidate = max(1.0, x_range * l_frac)
        for sf_frac in [0.5, 1.0, 2.0, 5.0]:
            sf_candidate = max(0.01, y_var * sf_frac)
            r = _np.abs(X[:, None] - X[None, :])
            s = sqrt5 * r / l_candidate
            K = sf_candidate * (1.0 + s + s * s / 3.0) * _np.exp(-s)
            Ky = K + _np.diag(R_diag) + jitter * _np.eye(n)
            try:
                L = _np.linalg.cholesky(Ky)
            except _np.linalg.LinAlgError as _error:
                continue
            y_c = Y - (beta0 + beta1 * X)
            alpha = _np.linalg.solve(L.T, _np.linalg.solve(L, y_c))
            lml = (
                -0.5 * _coerce_float(y_c @ alpha)
                - _coerce_float(_np.sum(_np.log(_np.diag(L))))
                - 0.5 * n * _math.log(2 * _math.pi)
            )
            if lml > best_lml:
                best_lml = lml
                best_l = l_candidate
                best_sf = sf_candidate

    return best_l, best_sf, n


def _matern52_kernel(
    x1: np.ndarray, x2: np.ndarray, sigma2_f: float, lengthscale: float
) -> np.ndarray:
    """Matérn-5/2 kernel matrix between two sets of 1D points."""
    r = np.abs(x1[:, None] - x2[None, :])
    s = math.sqrt(5.0) * r / lengthscale
    return sigma2_f * (1.0 + s + s * s / 3.0) * np.exp(-s)


def _matern52_product_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    sigma2_f: float,
    lengthscales: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """ARD product Matérn-5/2 kernel for d=2 companion models."""
    scales = np.asarray(lengthscales, dtype=float)
    distances = np.abs(x1[:, None, :] - x2[None, :, :])
    s = math.sqrt(5.0) * distances / scales[None, None, :]
    factors = (1.0 + s + s * s / 3.0) * np.exp(-s)
    return sigma2_f * np.prod(factors, axis=2)


def _gp_hp_grid_search_2d(
    points: list[tuple[float, float]],
    ys: list[float],
    rs: list[float],
    beta0: float,
    beta: tuple[float, float],
    lengthscales: tuple[float, float],
    sigma2_f: float,
) -> tuple[tuple[float, float], float, int]:
    """Picklable coarse ARD search using the shared GP process pool."""
    n = len(points)
    if n < _GP_HP_OPT_MIN_N:
        return lengthscales, sigma2_f, n
    X = np.asarray(points, dtype=float)
    Y = np.asarray(ys, dtype=float)
    R_diag = np.asarray(rs, dtype=float)
    ranges = np.ptp(X, axis=0) if n > 1 else np.asarray(lengthscales)
    ranges = np.maximum(ranges, 1.0)
    y_var = np.var(Y).item() if n > 1 else 1.0
    fractions = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    best_lml = -math.inf
    best_l = lengthscales
    best_sf = sigma2_f
    centered = Y - (beta0 + X @ np.asarray(beta, dtype=float))
    for frac_x in fractions:
        for frac_k in fractions:
            candidate = (
                max(1e-3, _coerce_float(ranges[0]) * frac_x),
                max(1e-3, _coerce_float(ranges[1]) * frac_k),
            )
            for sf_frac in (0.5, 1.0, 2.0, 5.0):
                sf_candidate = max(0.01, y_var * sf_frac)
                K = _matern52_product_kernel(X, X, sf_candidate, candidate)
                Ky = K + np.diag(R_diag) + _GP_JITTER * np.eye(n)
                try:
                    L = np.linalg.cholesky(Ky)
                except np.linalg.LinAlgError as _error:
                    continue
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, centered))
                lml = (
                    -0.5 * _coerce_float(centered @ alpha)
                    - _coerce_float(np.sum(np.log(np.diag(L))))
                    - 0.5 * n * math.log(2 * math.pi)
                )
                if lml > best_lml:
                    best_lml = lml
                    best_l = candidate
                    best_sf = sf_candidate
    return best_l, best_sf, n


@dataclass(frozen=True)
class _FitState:
    """ fix — atomic snapshot of every quantity needed by
    ``GPEstimate.predict``/``predict_grid``.  Bundling them prevents the
    background-fit / live-update race that surfaced as
    ``matmul: ... size 30 is different from 31`` in : previously
    ``self._L`` and ``self._alpha`` could be sized for an old training
    set while ``self._xs`` had already grown by one observation.

    ``epoch`` is the value of ``GPEstimate._fit_epoch`` at the time the
    fit was scheduled.  ``predict()`` rejects a fit-state whose epoch no
    longer matches and falls back to the prior mean — same Stage C /
    Step 7 D semantic that previously relied on ``self._L = None``.
    """

    L: np.ndarray
    alpha: np.ndarray
    xs: np.ndarray
    beta0: float
    beta1: float
    sigma2_f: float
    lengthscale: float
    epoch: int
    beta: np.ndarray | None = None
    lengthscales: np.ndarray | None = None
    input_dim: int = 1


class GPEstimate:
    """Gaussian Process estimator for resource usage as a function of input_size.

    Matérn-5/2 kernel + linear mean function, heteroscedastic noise,
    sliding window eviction.  See <docs> .5.3.

    Usage::

        gp = GPEstimate()
        gp.update(x=58.0, y=0.45, R=0.01)   # solo observation
        gp.update(x=180.0, y=0.82, R=0.05)  # corrected concurrent
        mu, var = gp.predict(100.0)          # unseen input_size
    """

    _main_loop: ClassVar[asyncio.AbstractEventLoop | None] = None

    def __init__(
        self,
        sigma2_f: float = 1.0,
        lengthscale: float = 50.0,
        n_max: int = _GP_N_MAX,
        max_age: float = _GP_MAX_AGE,
        input_dim: int = 1,
    ):
        if input_dim not in {1, 2}:
            raise ValueError("GPEstimate input_dim must be 1 or 2")
        self.input_dim = input_dim
        self.sigma2_f = sigma2_f
        self.lengthscale = lengthscale
        self.lengthscales: tuple[float, float] = (
            _coerce_float(lengthscale),
            _coerce_float(lengthscale),
        )

        self.beta0: float = 0.0
        self.beta1: float = 0.0
        self.beta: tuple[float, float] = (0.0, 0.0)
        self._mean_fitted: bool = False

        self.n_max = n_max
        self.max_age = max_age

        self._xs: list[float] = []
        self._xs_bounds_cache: tuple[int, tuple[float, float] | None] | None = None
        self._points: list[tuple[float, float]] = []
        self._ys: list[float] = []
        self._rs: list[float] = []
        self._ts: list[float] = []

        self._fit_state: _FitState | None = None
        self._fit_epoch: int = 0

        self._as_dict_cache: dict[str, Any] | None = None

        self._n_total: int = 0

        self._fit_inflight: bool = False
        self._fit_reschedule_requested: bool = False
        self._hp_inflight: bool = False
        self._hp_reschedule_requested: bool = False
        self._fit_loop: asyncio.AbstractEventLoop | None = None

    @property
    def n(self) -> int:
        """Number of observations in the current window."""
        return len(self._points) if self.input_dim == 2 else len(self._xs)

    def _mean_fn(self, x: np.ndarray) -> np.ndarray:
        return self.beta0 + self.beta1 * x

    def _invalidate_fit(self) -> None:
        """ fix — bump epoch + clear cached fit-state.

        Called from every site that mutates ``self._xs`` / ``self._ys``
        / ``self._rs`` / ``self.beta0`` / ``self.beta1`` / ``self.sigma2_f``
        / ``self.lengthscale``.  An in-flight background fit will see the
        epoch mismatch on completion and discard its result instead of
        overwriting a fresher fit-state.
        """
        self._fit_epoch += 1
        self._fit_state = None
        self._xs_bounds_cache = None

    def _evict_old(self) -> None:
        """Remove observations older than max_age or exceeding n_max."""
        now = time.monotonic()
        invalidated = False
        cutoff = now - self.max_age
        keep = [i for i, t in enumerate(self._ts) if t >= cutoff]
        if len(keep) < len(self._xs):
            self._xs = [self._xs[i] for i in keep]
            self._ys = [self._ys[i] for i in keep]
            self._rs = [self._rs[i] for i in keep]
            self._ts = [self._ts[i] for i in keep]
            invalidated = True

        while len(self._xs) > self.n_max:
            self._xs.pop(0)
            self._ys.pop(0)
            self._rs.pop(0)
            self._ts.pop(0)
            invalidated = True

        if invalidated:
            self._invalidate_fit()

    def _fit_cholesky(self) -> None:
        """Recompute Cholesky factor L of [K + R + jitter*I] synchronously.

        Used only on the test / pre-loop bootstrap path where ``update()``
        cannot schedule onto an asyncio executor.  Production main-loop
        path goes through ``_fit_cholesky_from_snapshot``.
        """
        if self.n == 0:
            self._fit_state = None
            return

        xs_snap = np.array(self._xs)
        ys_snap = np.array(self._ys)
        rs_snap = np.array(self._rs)
        beta0 = self.beta0
        beta1 = self.beta1
        sigma2_f = self.sigma2_f
        lengthscale = self.lengthscale
        epoch_at_schedule = self._fit_epoch

        K = _matern52_kernel(xs_snap, xs_snap, sigma2_f, lengthscale)
        R = np.diag(rs_snap)
        Ky = K + R + _GP_JITTER * np.eye(len(xs_snap))

        try:
            L_new = np.linalg.cholesky(Ky)
        except np.linalg.LinAlgError as _error:
            Ky += 1e-4 * np.eye(len(xs_snap))
            L_new = np.linalg.cholesky(Ky)

        y_centered = ys_snap - (beta0 + beta1 * xs_snap)
        alpha_new = np.linalg.solve(L_new.T, np.linalg.solve(L_new, y_centered))

        self._fit_state = _FitState(
            L=L_new,
            alpha=alpha_new,
            xs=xs_snap,
            beta0=beta0,
            beta1=beta1,
            sigma2_f=sigma2_f,
            lengthscale=lengthscale,
            epoch=epoch_at_schedule,
        )

    def _fit_mean_function(self) -> None:
        """Fit linear mean function m(x) = beta0 + beta1*x via least squares.

        Plan fix — Fix n=1 degenerate case that left the
        mean function at its default (0, 0), causing the posterior to
        shrink an isolated observation heavily toward the zero prior.
        The symptom manifested most visibly in ``InitProfile`` where
        every observation is at x=0 and Case-3 interference inflates R
        to ``(0.5 × raw)²`` (≈2500 for protenix 100 s init): a single
        per-GPU observation reported μ* ≈ σ²_f·y/(σ²_f+R) ≈ 100·100/2600
        ≈ 3.85 s (ops console showed 4.0 s) while the cross-GPU pool
        with n=4 hit the ``x_unique < 2`` branch below, set beta0 to
        mean(Y) ≈ 100 s, and correctly reported ~97 s.  The phase
        transition at n=2 was an observable bug in the init-latency
        table.  Fix: for n=1, seed the mean function with the single
        observation (beta0 = Y[0], beta1 = 0) so the GP does not shrink
        toward an unrelated zero prior.  With n ≥ 2 the existing
        least-squares / same-x fallback branches take over.
        """
        if self.n == 0:
            return
        if self.n == 1:
            self.beta0 = _coerce_float(self._ys[0])
            self.beta1 = 0.0
            self._mean_fitted = True
            return
        X = np.array(self._xs)
        Y = np.array(self._ys)
        x_unique = len({round(x, 1) for x in self._xs})
        if x_unique < 2:
            self.beta0 = _coerce_float(np.mean(Y))
            self.beta1 = 0.0
            self._mean_fitted = True
            return
        x_mean = np.mean(X)
        y_mean = np.mean(Y)
        ss_xx = np.sum((X - x_mean) ** 2)
        if ss_xx < 1e-12:
            self.beta0 = _coerce_float(y_mean)
            self.beta1 = 0.0
        else:
            self.beta1 = _coerce_float(np.sum((X - x_mean) * (Y - y_mean)) / ss_xx)
            self.beta0 = _coerce_float(y_mean - self.beta1 * x_mean)
        self._mean_fitted = True

    def _optimize_hyperparameters(self) -> None:
        """Simple hyperparameter optimization via grid search on log-marginal likelihood.

        Full L-BFGS optimization is overkill for n < 200; a coarse grid
        over lengthscale candidates is sufficient and robust.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            fallback = self._fit_loop or GPEstimate._main_loop
            if fallback is not None and not fallback.is_closed():
                if self.n < _GP_HP_OPT_MIN_N:
                    return
                if self._hp_inflight:
                    self._hp_reschedule_requested = True
                    return
                self._hp_inflight = True
                xs_snap = list(self._xs)
                ys_snap = list(self._ys)
                rs_snap = list(self._rs)
                beta0_snap = self.beta0
                beta1_snap = self.beta1
                lengthscale_snap = self.lengthscale
                sigma2_f_snap = self.sigma2_f
                epoch_snap = self._fit_epoch
                fallback.call_soon_threadsafe(
                    self._dispatch_optimize_hyperparameters,
                    xs_snap,
                    ys_snap,
                    rs_snap,
                    beta0_snap,
                    beta1_snap,
                    lengthscale_snap,
                    sigma2_f_snap,
                    epoch_snap,
                )
                return
            return
        if self.n < _GP_HP_OPT_MIN_N:
            return
        self._fit_loop = loop
        if GPEstimate._main_loop is None:
            GPEstimate._main_loop = loop
        if self._hp_inflight:
            self._hp_reschedule_requested = True
            return
        self._hp_inflight = True
        xs_snap = list(self._xs)
        ys_snap = list(self._ys)
        rs_snap = list(self._rs)
        beta0_snap = self.beta0
        beta1_snap = self.beta1
        lengthscale_snap = self.lengthscale
        sigma2_f_snap = self.sigma2_f
        epoch_snap = self._fit_epoch
        future = loop.run_in_executor(
            _get_gp_process_pool(),
            _gp_hp_grid_search,
            xs_snap,
            ys_snap,
            rs_snap,
            beta0_snap,
            beta1_snap,
            lengthscale_snap,
            sigma2_f_snap,
        )
        future.add_done_callback(lambda f: self._on_hp_grid_search_done(f, epoch_snap))
        return

    def _dispatch_optimize_hyperparameters(
        self,
        xs: list[float],
        ys: list[float],
        rs: list[float],
        beta0: float,
        beta1: float,
        lengthscale: float,
        sigma2_f: float,
        epoch_at_schedule: int,
    ) -> None:
        """Legacy cross-thread dispatch path (fallback loop).  fix:
        routes through ProcessPoolExecutor just like the main path."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        future = loop.run_in_executor(
            _get_gp_process_pool(),
            _gp_hp_grid_search,
            xs,
            ys,
            rs,
            beta0,
            beta1,
            lengthscale,
            sigma2_f,
        )
        future.add_done_callback(
            lambda f: self._on_hp_grid_search_done(f, epoch_at_schedule)
        )

    def _on_hp_grid_search_done(
        self,
        future: asyncio.Future[tuple[float, float, int]],
        epoch_at_schedule: int,
    ) -> None:
        """ProcessPoolExecutor done-callback (fix).

        Called from the asyncio event loop thread (asyncio wraps the
        concurrent.futures Future and invokes add_done_callback on the
        loop thread).  We therefore access ``self`` state safely without
        locks.
        """
        import time as _time

        _hp_start = _time.time()
        try:
            exc = future.exception()
            if exc is not None:
                _LOG.warning("[hot-path] GPEstimate._gp_hp_grid_search raised: %s", exc)
                return
            best_l, best_sf, n = future.result()
            if epoch_at_schedule != self._fit_epoch:
                return
            self.lengthscale = best_l
            self.sigma2_f = best_sf
            self._invalidate_fit()
            _hp_elapsed_ms = (_time.time() - _hp_start) * 1000.0
            if _hp_elapsed_ms > 5.0:
                _LOG.warning(
                    "[hot-path] GPEstimate._on_hp_grid_search_done n=%d "
                    "callback_elapsed=%.1fms",
                    n,
                    _hp_elapsed_ms,
                )
        finally:
            self._hp_inflight = False
            if self._hp_reschedule_requested:
                self._hp_reschedule_requested = False
                loop = self._fit_loop
                if loop is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(self._optimize_hyperparameters)

    def _optimize_hyperparameters_worker(
        self,
        xs: list[float],
        ys: list[float],
        rs: list[float],
        beta0: float,
        beta1: float,
        lengthscale: float,
        sigma2_f: float,
        epoch_at_schedule: int,
    ) -> None:
        """Kept for test compatibility; production path uses ProcessPoolExecutor."""
        try:
            best_l, best_sf, _n = _gp_hp_grid_search(
                xs, ys, rs, beta0, beta1, lengthscale, sigma2_f
            )
            if epoch_at_schedule == self._fit_epoch:
                self.lengthscale = best_l
                self.sigma2_f = best_sf
                self._invalidate_fit()
        finally:
            self._hp_inflight = False
            if self._hp_reschedule_requested:
                self._hp_reschedule_requested = False
                loop = self._fit_loop
                if loop is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(self._optimize_hyperparameters)

    def _optimize_hyperparameters_from_snapshot(
        self,
        xs: list[float],
        ys: list[float],
        rs: list[float],
        beta0: float,
        beta1: float,
        lengthscale: float,
        sigma2_f: float,
        epoch_at_schedule: int,
    ) -> None:
        """Synchronous fallback (legacy test path / impl reference).

        fix: production HP opt now goes through
        ``_gp_hp_grid_search`` in a ProcessPoolExecutor.  This method
        is retained so existing unit tests that call it directly still
        pass without modification.
        """
        import time as _time

        _hp_start = _time.time()
        try:
            best_l, best_sf, n = _gp_hp_grid_search(
                xs, ys, rs, beta0, beta1, lengthscale, sigma2_f
            )
            if epoch_at_schedule != self._fit_epoch:
                return
            self.lengthscale = best_l
            self.sigma2_f = best_sf
            self._invalidate_fit()
        finally:
            _hp_elapsed_ms = (_time.time() - _hp_start) * 1000.0
            if _hp_elapsed_ms > 20.0:
                _LOG.warning(
                    "[hot-path] GPEstimate._optimize_hyperparameters "
                    "background n=%d elapsed=%.1fms",
                    len(xs),
                    _hp_elapsed_ms,
                )

    def _optimize_hyperparameters_impl(self) -> None:
        if self.n < _GP_HP_OPT_MIN_N:
            return

        X = np.array(self._xs)
        Y = np.array(self._ys)
        R_diag = np.array(self._rs)

        x_range = _coerce_float(np.max(X) - np.min(X)) if len(X) > 1 else 100.0
        y_var = _coerce_float(np.var(Y)) if len(Y) > 1 else 1.0

        best_lml = -_coerce_float("inf")
        best_l = self.lengthscale
        best_sf = self.sigma2_f

        for l_frac in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            l_candidate = max(1.0, x_range * l_frac)
            for sf_frac in [0.5, 1.0, 2.0, 5.0]:
                sf_candidate = max(0.01, y_var * sf_frac)
                K = _matern52_kernel(X, X, sf_candidate, l_candidate)
                Ky = K + np.diag(R_diag) + _GP_JITTER * np.eye(self.n)
                try:
                    L = np.linalg.cholesky(Ky)
                except np.linalg.LinAlgError as _error:
                    continue
                y_c = Y - (self.beta0 + self.beta1 * X)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_c))
                lml = (
                    -0.5 * _coerce_float(y_c @ alpha)
                    - _coerce_float(np.sum(np.log(np.diag(L))))
                    - 0.5 * self.n * math.log(2 * math.pi)
                )
                if lml > best_lml:
                    best_lml = lml
                    best_l = l_candidate
                    best_sf = sf_candidate

        self.lengthscale = best_l
        self.sigma2_f = best_sf
        self._invalidate_fit()

    def _fit_2d_mean(self) -> None:
        if not self._points:
            return
        X = np.asarray(self._points, dtype=float)
        Y = np.asarray(self._ys, dtype=float)
        design = np.column_stack((np.ones(len(X)), X))
        if len(X) >= 3 and np.linalg.matrix_rank(design) == 3:
            coefficients, *_ = np.linalg.lstsq(design, Y, rcond=None)
            self.beta0 = _coerce_float(coefficients[0])
            self.beta = (_coerce_float(coefficients[1]), _coerce_float(coefficients[2]))
        else:
            self.beta0 = _coerce_float(np.mean(Y))
            self.beta = (0.0, 0.0)
        self._mean_fitted = True

    def _fit_2d_from_snapshot(
        self,
        points: list[tuple[float, float]],
        ys: list[float],
        rs: list[float],
        beta0: float,
        beta: tuple[float, float],
        sigma2_f: float,
        lengthscales: tuple[float, float],
        epoch_at_schedule: int,
    ) -> None:
        if not points:
            return
        try:
            X = np.asarray(points, dtype=float)
            Y = np.asarray(ys, dtype=float)
            K = _matern52_product_kernel(X, X, sigma2_f, lengthscales)
            Ky = K + np.diag(np.asarray(rs, dtype=float)) + _GP_JITTER * np.eye(len(X))
            try:
                L = np.linalg.cholesky(Ky)
            except np.linalg.LinAlgError as _error:
                L = np.linalg.cholesky(Ky + 1e-4 * np.eye(len(X)))
            beta_array = np.asarray(beta, dtype=float)
            centered = Y - (beta0 + X @ beta_array)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, centered))
            if epoch_at_schedule != self._fit_epoch:
                return
            self._fit_state = _FitState(
                L=L,
                alpha=alpha,
                xs=X,
                beta0=beta0,
                beta1=0.0,
                sigma2_f=sigma2_f,
                lengthscale=_coerce_float(lengthscales[0]),
                epoch=epoch_at_schedule,
                beta=beta_array,
                lengthscales=np.asarray(lengthscales, dtype=float),
                input_dim=2,
            )
        except Exception as _error:
            _LOG.debug("[gp-stage-c] d=2 background fit failed", exc_info=True)

    def _fit_2d_worker(self, *snapshot: Any) -> None:
        try:
            self._fit_2d_from_snapshot(*snapshot)
        finally:
            self._fit_inflight = False
            if self._fit_reschedule_requested:
                self._fit_reschedule_requested = False
                loop = self._fit_loop
                if loop is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(self._schedule_2d_fit)

    def _fit_2d_snapshot(self) -> tuple[Any, ...]:
        return (
            list(self._points),
            list(self._ys),
            list(self._rs),
            self.beta0,
            self.beta,
            self.sigma2_f,
            self.lengthscales,
            self._fit_epoch,
        )

    def _schedule_2d_fit(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._fit_2d_from_snapshot(*self._fit_2d_snapshot())
            return
        self._fit_loop = loop
        if self._fit_inflight:
            self._fit_reschedule_requested = True
            return
        self._fit_inflight = True
        loop.run_in_executor(None, self._fit_2d_worker, *self._fit_2d_snapshot())

    def _on_2d_hp_done(
        self,
        future: asyncio.Future[tuple[tuple[float, float], float, int]],
        epoch_at_schedule: int,
    ) -> None:
        try:
            if future.exception() is not None:
                _LOG.warning("d=2 GP hyperparameter search failed", exc_info=True)
                return
            lengthscales, sigma2_f, _ = future.result()
            if epoch_at_schedule != self._fit_epoch:
                return
            self.lengthscales = (
                _coerce_float(lengthscales[0]),
                _coerce_float(lengthscales[1]),
            )
            self.sigma2_f = _coerce_float(sigma2_f)
            self._invalidate_fit()
        finally:
            self._hp_inflight = False
            if self._hp_reschedule_requested:
                self._hp_reschedule_requested = False
                loop = self._fit_loop
                if loop is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(self._optimize_2d_hyperparameters)

    def _optimize_2d_hyperparameters(self) -> None:
        if self.n < _GP_HP_OPT_MIN_N or self._hp_inflight:
            self._hp_reschedule_requested = self._hp_inflight
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._fit_loop = loop
        self._hp_inflight = True
        epoch = self._fit_epoch
        future = loop.run_in_executor(
            _get_gp_process_pool(),
            _gp_hp_grid_search_2d,
            list(self._points),
            list(self._ys),
            list(self._rs),
            self.beta0,
            self.beta,
            self.lengthscales,
            self.sigma2_f,
        )
        future.add_done_callback(lambda done: self._on_2d_hp_done(done, epoch))

    def _update_2d(self, point: Any, y: float, R: float) -> None:
        try:
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError
            point_x, point_k = _coerce_float(point[0]), _coerce_float(point[1])
            target, noise = _coerce_float(y), _coerce_float(R)
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                "d=2 GP observation must contain numeric (x, k), y and R"
            ) from exc
        if (
            not math.isfinite(point_x)
            or not math.isfinite(point_k)
            or point_x <= 0
            or point_k <= 0
        ):
            raise ValueError("d=2 GP points must be positive finite (x, k) pairs")
        if not math.isfinite(target) or not math.isfinite(noise) or noise < 0:
            raise ValueError("d=2 GP target/noise must be finite with R >= 0")
        self._points.append((point_x, point_k))
        self._ys.append(target)
        self._rs.append(noise)
        self._ts.append(time.monotonic())
        self._n_total += 1
        self._invalidate_fit()
        self._as_dict_cache = None

        cutoff = time.monotonic() - self.max_age
        keep = [index for index, ts in enumerate(self._ts) if ts >= cutoff]
        if len(keep) != len(self._points):
            self._points = [self._points[index] for index in keep]
            self._ys = [self._ys[index] for index in keep]
            self._rs = [self._rs[index] for index in keep]
            self._ts = [self._ts[index] for index in keep]
            self._invalidate_fit()
        while len(self._points) > self.n_max:
            self._points.pop(0)
            self._ys.pop(0)
            self._rs.pop(0)
            self._ts.pop(0)
            self._invalidate_fit()
        self._fit_2d_mean()
        if self._n_total % _GP_HP_OPT_INTERVAL == 0:
            self._optimize_2d_hyperparameters()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            self._fit_loop = loop
            if GPEstimate._main_loop is None:
                GPEstimate._main_loop = loop
            self._schedule_2d_fit()
            return
        fallback = self._fit_loop or GPEstimate._main_loop
        if fallback is not None and not fallback.is_closed():
            fallback.call_soon_threadsafe(self._schedule_2d_fit)
            return
        self._fit_2d_from_snapshot(*self._fit_2d_snapshot())

    def update(self, x: Any, y: float, R: float) -> None:
        """Add an observation and refit.

        Parameters
        ----------
        x : input_size
        y : corrected observation value
        R : measurement noise variance (interference-aware)
        """
        if self.input_dim == 2:
            self._update_2d(x, y, R)
            return
        self._xs.append(x)
        self._ys.append(y)
        self._rs.append(R)
        self._ts.append(time.monotonic())
        self._n_total += 1
        self._invalidate_fit()
        self._as_dict_cache: dict[str, Any] | None = None

        self._evict_old()

        self._fit_mean_function()

        if self._n_total % _GP_HP_OPT_INTERVAL == 0 and self.n >= _GP_HP_OPT_MIN_N:
            self._optimize_hyperparameters()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            self._fit_loop = loop
            if GPEstimate._main_loop is None:
                GPEstimate._main_loop = loop
            if self._fit_inflight:
                self._fit_reschedule_requested = True
                return
            self._fit_inflight = True
            xs_snap = list(self._xs)
            ys_snap = list(self._ys)
            rs_snap = list(self._rs)
            beta0_snap = self.beta0
            beta1_snap = self.beta1
            sigma2_f_snap = self.sigma2_f
            lengthscale_snap = self.lengthscale
            epoch_snap = self._fit_epoch
            loop.run_in_executor(
                None,
                self._fit_cholesky_worker,
                xs_snap,
                ys_snap,
                rs_snap,
                beta0_snap,
                beta1_snap,
                sigma2_f_snap,
                lengthscale_snap,
                epoch_snap,
            )
            return
        fallback = self._fit_loop or GPEstimate._main_loop
        if fallback is not None and not fallback.is_closed():
            if self._fit_inflight:
                self._fit_reschedule_requested = True
                return
            self._fit_inflight = True
            xs_snap = list(self._xs)
            ys_snap = list(self._ys)
            rs_snap = list(self._rs)
            beta0_snap = self.beta0
            beta1_snap = self.beta1
            sigma2_f_snap = self.sigma2_f
            lengthscale_snap = self.lengthscale
            epoch_snap = self._fit_epoch
            fallback.call_soon_threadsafe(
                self._dispatch_cholesky_fit,
                xs_snap,
                ys_snap,
                rs_snap,
                beta0_snap,
                beta1_snap,
                sigma2_f_snap,
                lengthscale_snap,
                epoch_snap,
            )
            return
        self._fit_cholesky()

    def _dispatch_cholesky_fit(
        self,
        xs: list[float],
        ys: list[float],
        rs: list[float],
        beta0: float,
        beta1: float,
        sigma2_f: float,
        lengthscale: float,
        epoch_at_schedule: int,
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.run_in_executor(
            None,
            self._fit_cholesky_worker,
            xs,
            ys,
            rs,
            beta0,
            beta1,
            sigma2_f,
            lengthscale,
            epoch_at_schedule,
        )

    def _fit_cholesky_worker(
        self,
        xs: list[float],
        ys: list[float],
        rs: list[float],
        beta0: float,
        beta1: float,
        sigma2_f: float,
        lengthscale: float,
        epoch_at_schedule: int,
    ) -> None:
        try:
            self._fit_cholesky_from_snapshot(
                xs,
                ys,
                rs,
                beta0,
                beta1,
                sigma2_f,
                lengthscale,
                epoch_at_schedule,
            )
        finally:
            self._fit_inflight = False
            if self._fit_reschedule_requested:
                self._fit_reschedule_requested = False
                loop = self._fit_loop
                if loop is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(self._schedule_cholesky_fit)

    def _schedule_cholesky_fit(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._fit_cholesky()
            return
        self._fit_loop = loop
        if self._fit_inflight:
            self._fit_reschedule_requested = True
            return
        self._fit_inflight = True
        xs_snap = list(self._xs)
        ys_snap = list(self._ys)
        rs_snap = list(self._rs)
        beta0_snap = self.beta0
        beta1_snap = self.beta1
        sigma2_f_snap = self.sigma2_f
        lengthscale_snap = self.lengthscale
        epoch_snap = self._fit_epoch
        loop.run_in_executor(
            None,
            self._fit_cholesky_worker,
            xs_snap,
            ys_snap,
            rs_snap,
            beta0_snap,
            beta1_snap,
            sigma2_f_snap,
            lengthscale_snap,
            epoch_snap,
        )

    def _fit_cholesky_from_snapshot(
        self,
        xs: list[float],
        ys: list[float],
        rs: list[float],
        beta0: float,
        beta1: float,
        sigma2_f: float,
        lengthscale: float,
        epoch_at_schedule: int,
    ) -> None:
        """Background-thread companion to ``_fit_cholesky`` (fix).

        Operates on a snapshot of every input the fit consumes (training
        xs/ys/rs **and** mean-fn coefficients **and** kernel hyperparams),
        captured at ``update()`` time.  The completed fit is installed as
        a single ``_FitState`` (atomic Python attribute write under the
        GIL) only if ``epoch_at_schedule == self._fit_epoch`` at the
        install site — this CAS-like guard discards the result of any
        fit that has been superseded by a more recent ``update()`` /
        ``_evict_old()`` / ``_optimize_hyperparameters()``.

        ``predict()`` performs a second epoch check against the installed
        state, so even a stale install that races past this guard
        (extremely narrow window between the epoch read and the
        attribute write) is rejected on the next predict and the next
        background fit self-heals.
        """
        n = len(xs)
        if n == 0:
            return
        try:
            X = np.array(xs)
            K = _matern52_kernel(X, X, sigma2_f, lengthscale)
            R = np.diag(np.array(rs))
            Ky = K + R + _GP_JITTER * np.eye(n)
            try:
                L_new = np.linalg.cholesky(Ky)
            except np.linalg.LinAlgError as _error:
                Ky += 1e-4 * np.eye(n)
                L_new = np.linalg.cholesky(Ky)
            Y = np.array(ys)
            y_centered = Y - (beta0 + beta1 * X)
            alpha_new = np.linalg.solve(L_new.T, np.linalg.solve(L_new, y_centered))

            if epoch_at_schedule != self._fit_epoch:
                return
            self._fit_state = _FitState(
                L=L_new,
                alpha=alpha_new,
                xs=X,
                beta0=beta0,
                beta1=beta1,
                sigma2_f=sigma2_f,
                lengthscale=lengthscale,
                epoch=epoch_at_schedule,
            )
        except Exception as _error:
            _LOG.debug(
                "[gp-stage-c] background fit failed (n=%d), prior fallback in effect",
                n,
            )

    @staticmethod
    def _coerce_2d_point(point: Any) -> np.ndarray:
        try:
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError
            values = np.asarray(
                (_coerce_float(point[0]), _coerce_float(point[1])), dtype=float
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("d=2 GP prediction requires an (x, k) pair") from exc
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("d=2 GP prediction point must be positive and finite")
        return values

    def _predict_2d(self, point: Any) -> tuple[float, float]:
        values = self._coerce_2d_point(point)
        current_beta = np.asarray(self.beta, dtype=float)
        prior = _coerce_float(self.beta0 + values @ current_beta)
        if self.n == 0:
            return prior, _coerce_float(self.sigma2_f)
        local = self._fit_state
        if (
            local is None
            or local.epoch != self._fit_epoch
            or local.input_dim != 2
            or local.beta is None
            or local.lengthscales is None
        ):
            return prior, _coerce_float(self.sigma2_f)
        point_matrix = values.reshape(1, 2)
        k_star = _matern52_product_kernel(
            point_matrix,
            local.xs,
            local.sigma2_f,
            local.lengthscales,
        )[0]
        mean = _coerce_float(local.beta0 + values @ local.beta + k_star @ local.alpha)
        solved = np.linalg.solve(local.L, k_star)
        variance = max(
            _coerce_float(local.sigma2_f - solved @ solved),
            _GP_JITTER,
        )
        return mean, variance

    def predict(self, x_star: Any) -> tuple[float, float]:
        """Predict mean and variance at input_size x_star.

        Returns (mu*, sigma2*). If no observations, returns (mean_fn(x*), sigma2_f).

         fix — reads a single ``_FitState`` snapshot and
        uses **only** quantities from that snapshot (training xs, L,
        alpha, mean-fn coefficients, kernel hyperparams).  Live
        ``self._xs`` / ``self.beta0`` / ``self.lengthscale`` are NEVER
        touched.  This eliminates the  race where ``self._L`` was
        sized for the previous training set while ``self._xs`` had
        already grown by one observation, producing a
        ``matmul: ... size 30 is different from 31`` ValueError on
        every ``predict()`` call.

        Stage C / Step 7 D semantic preserved: when ``_fit_state`` is
        None or its epoch has been superseded (post-update,
        pre-background-refit) we return the prior mean instead of
        synchronously refitting on the calling thread.
        """
        if self.input_dim == 2:
            return self._predict_2d(x_star)
        if self.n == 0:
            return (self._mean_fn(np.array([x_star]))[0], self.sigma2_f)

        local = self._fit_state
        if local is None or local.epoch != self._fit_epoch:
            x_s = np.array([x_star])
            mu = _coerce_float(self._mean_fn(x_s)[0])
            return (mu, _coerce_float(self.sigma2_f))

        X = local.xs
        x_s = np.array([x_star])

        k_star = _matern52_kernel(x_s, X, local.sigma2_f, local.lengthscale)[0]

        mu_x_star = _coerce_float(local.beta0 + local.beta1 * _coerce_float(x_star))
        mu_star = mu_x_star + _coerce_float(k_star @ local.alpha)

        v = np.linalg.solve(local.L, k_star)
        sigma2_star = _coerce_float(local.sigma2_f - v @ v)
        sigma2_star = max(sigma2_star, _GP_JITTER)

        return (mu_star, sigma2_star)

    def predict_grid(self, x_grid: list[Any]) -> tuple[list[float], list[float]]:
        """Predict mean and variance at multiple input_sizes (batch).

        Returns (means, variances) lists.

         fix — same ``_FitState`` snapshot semantics as
        ``predict()``.  Stage C / Step 7 D stale fallback preserved.
        """
        if self.input_dim == 2:
            if not x_grid:
                return [], []
            points = np.vstack([self._coerce_2d_point(point) for point in x_grid])
            current_beta = np.asarray(self.beta, dtype=float)
            prior = self.beta0 + points @ current_beta
            local = self._fit_state
            if (
                self.n == 0
                or local is None
                or local.epoch != self._fit_epoch
                or local.input_dim != 2
                or local.beta is None
                or local.lengthscales is None
            ):
                return prior.tolist(), [_coerce_float(self.sigma2_f)] * len(points)
            kernel = _matern52_product_kernel(
                points,
                local.xs,
                local.sigma2_f,
                local.lengthscales,
            )
            means = local.beta0 + points @ local.beta + kernel @ local.alpha
            solved = np.linalg.solve(local.L, kernel.T)
            variances = np.maximum(
                local.sigma2_f - np.sum(solved * solved, axis=0),
                _GP_JITTER,
            )
            return means.tolist(), variances.tolist()
        if self.n == 0:
            xg = np.array(x_grid)
            m = self._mean_fn(xg)
            return (m.tolist(), [self.sigma2_f] * len(x_grid))

        local = self._fit_state
        if local is None or local.epoch != self._fit_epoch:
            Xs = np.array(x_grid)
            means = self._mean_fn(Xs)
            return (means.tolist(), [_coerce_float(self.sigma2_f)] * len(x_grid))

        X = local.xs
        Xs = np.array(x_grid)

        K_star = _matern52_kernel(Xs, X, local.sigma2_f, local.lengthscale)
        m_at_grid = local.beta0 + local.beta1 * Xs
        means = m_at_grid + K_star @ local.alpha

        V = np.linalg.solve(local.L, K_star.T)
        variances = local.sigma2_f - np.sum(V * V, axis=0)
        variances = np.maximum(variances, _GP_JITTER)

        return (means.tolist(), variances.tolist())

    @property
    def mean(self) -> float:
        """Cross-input mean (backward compat with GaussianEstimate)."""
        if self.n == 0:
            return 0.0
        return _coerce_float(np.mean(self._ys))

    @property
    def std(self) -> float:
        """Cross-input std (backward compat)."""
        if self.n < 2:
            return 0.0
        return _coerce_float(np.std(self._ys, ddof=0))

    def confidence(self, x: Any = None) -> float:
        """GP-native confidence: 1 - σ²*(x) / σ²_f.

        If x is None, returns average confidence across observed input_sizes.
        """
        if self.n == 0:
            return 0.0
        if x is not None:
            _, var = self.predict(x)
            raw = 1.0 - var / self.sigma2_f
            return round(max(0.0, min(1.0, raw)), 3)
        observed = self._points if self.input_dim == 2 else self._xs
        _, variances = self.predict_grid(observed)
        raw = 1.0 - np.mean(variances) / self.sigma2_f
        return round(max(0.0, min(1.0, _coerce_float(raw))), 3)

    def is_sufficient(self, x: Any, required_relative_error: float) -> bool:
        """CRLB-backed precision check at input_size x.

        sufficient(ε, x) ⟺ 1.96 × √σ²*(x) / |μ*(x)| ≤ ε
        """
        mu, var = self.predict(x)
        if abs(mu) < 1e-9:
            return 1.96 * math.sqrt(var) < 1.0
        return 1.96 * math.sqrt(var) <= required_relative_error * abs(mu)

    def relative_error(self, x: Any) -> float:
        """Current relative estimation error at input_size x."""
        mu, var = self.predict(x)
        if abs(mu) < 1e-9:
            return _coerce_float("inf")
        return 1.96 * math.sqrt(var) / abs(mu)

    def has_full_rank_support(self) -> bool:
        if self.input_dim != 2 or self.n < 3:
            return False
        design = np.column_stack(
            (np.ones(self.n), np.asarray(self._points, dtype=float))
        )
        return (
            np.linalg.matrix_rank(design) == 3
            and len({point[0] for point in self._points}) >= 2
            and len({point[1] for point in self._points}) >= 2
        )

    def export_state(self) -> dict[str, Any]:
        """Export raw GP state for exact completed-run bootstrap."""
        now = time.monotonic()
        if self.input_dim == 2:
            return {
                "schema": "gp_estimate_v2",
                "input_dim": 2,
                "sigma2_f": _coerce_float(self.sigma2_f),
                "lengthscales": [_coerce_float(value) for value in self.lengthscales],
                "beta0": _coerce_float(self.beta0),
                "beta": [_coerce_float(value) for value in self.beta],
                "mean_fitted": bool(self._mean_fitted),
                "n_max": _coerce_int(self.n_max),
                "max_age": _coerce_float(self.max_age),
                "n_total": _coerce_int(self._n_total),
                "fit_epoch": _coerce_int(self._fit_epoch),
                "observations": [
                    {
                        "point": [_coerce_float(point[0]), _coerce_float(point[1])],
                        "y": _coerce_float(self._ys[index]),
                        "R": _coerce_float(self._rs[index]),
                        "age_sec": max(0.0, now - _coerce_float(self._ts[index])),
                    }
                    for index, point in enumerate(self._points)
                ],
            }
        return {
            "schema": "gp_estimate_v1",
            "sigma2_f": _coerce_float(self.sigma2_f),
            "lengthscale": _coerce_float(self.lengthscale),
            "beta0": _coerce_float(self.beta0),
            "beta1": _coerce_float(self.beta1),
            "mean_fitted": bool(self._mean_fitted),
            "n_max": _coerce_int(self.n_max),
            "max_age": _coerce_float(self.max_age),
            "n_total": _coerce_int(self._n_total),
            "fit_epoch": _coerce_int(self._fit_epoch),
            "observations": [
                {
                    "x": _coerce_float(x),
                    "y": _coerce_float(self._ys[index]),
                    "R": _coerce_float(self._rs[index]),
                    "age_sec": max(0.0, now - _coerce_float(self._ts[index])),
                }
                for index, x in enumerate(self._xs)
            ],
        }

    def import_state(self, data: Mapping[str, Any]) -> None:
        """Replace raw GP state from an ``export_state`` payload."""
        self._xs_bounds_cache = None
        now = time.monotonic()
        observations = data.get("observations") or []
        if self.input_dim == 2:
            self._points = []
            self._ys = []
            self._rs = []
            self._ts = []
            self._fit_state = None
            self._as_dict_cache = None
            if data.get("schema") != "gp_estimate_v2":
                self._n_total = 0
                self._fit_epoch = 0
                return
            try:
                scales = list(data.get("lengthscales") or self.lengthscales)
                slopes = list(data.get("beta") or self.beta)
                if len(scales) != 2 or len(slopes) != 2:
                    raise ValueError
                self.lengthscales = (_coerce_float(scales[0]), _coerce_float(scales[1]))
                self.beta = (_coerce_float(slopes[0]), _coerce_float(slopes[1]))
                self.sigma2_f = _coerce_float(data.get("sigma2_f", self.sigma2_f))
                self.beta0 = _coerce_float(data.get("beta0", 0.0) or 0.0)
                self._mean_fitted = bool(data.get("mean_fitted", False))
                self.n_max = _coerce_int(data.get("n_max", self.n_max) or self.n_max)
                self.max_age = _coerce_float(
                    data.get("max_age", self.max_age) or self.max_age
                )
                self._n_total = _coerce_int(data.get("n_total", len(observations)) or 0)
                self._fit_epoch = _coerce_int(data.get("fit_epoch", 0) or 0)
            except (TypeError, ValueError, IndexError):
                self._n_total = 0
                self._fit_epoch = 0
                return
            for item in observations if isinstance(observations, list) else []:
                if not isinstance(item, Mapping):
                    continue
                try:
                    point = list(item.get("point") or [])
                    point_x, point_k = _coerce_float(point[0]), _coerce_float(point[1])
                    target = _coerce_float(item.get("y"))
                    noise = _coerce_float(item.get("R"))
                    age = max(0.0, _coerce_float(item.get("age_sec", 0.0) or 0.0))
                except (TypeError, ValueError, IndexError):
                    continue
                if (
                    point_x <= 0
                    or point_k <= 0
                    or noise < 0
                    or not all(
                        math.isfinite(value)
                        for value in (point_x, point_k, target, noise)
                    )
                ):
                    continue
                self._points.append((point_x, point_k))
                self._ys.append(target)
                self._rs.append(noise)
                self._ts.append(now - age)
            self._fit_inflight = False
            self._fit_reschedule_requested = False
            self._hp_inflight = False
            self._hp_reschedule_requested = False
            if self.n > 0:
                self._fit_2d_from_snapshot(*self._fit_2d_snapshot())
            return
        self.sigma2_f = _coerce_float(
            data.get("sigma2_f", self.sigma2_f) or self.sigma2_f
        )
        self.lengthscale = _coerce_float(
            data.get("lengthscale", self.lengthscale) or self.lengthscale
        )
        self.beta0 = _coerce_float(data.get("beta0", 0.0) or 0.0)
        self.beta1 = _coerce_float(data.get("beta1", 0.0) or 0.0)
        self._mean_fitted = bool(data.get("mean_fitted", False))
        self.n_max = _coerce_int(data.get("n_max", self.n_max) or self.n_max)
        self.max_age = _coerce_float(data.get("max_age", self.max_age) or self.max_age)
        self._n_total = _coerce_int(data.get("n_total", len(observations)) or 0)
        self._fit_epoch = _coerce_int(data.get("fit_epoch", 0) or 0)
        self._xs = []
        self._ys = []
        self._rs = []
        self._ts = []
        for item in observations if isinstance(observations, list) else []:
            if not isinstance(item, Mapping):
                continue
            try:
                x = _coerce_float(item.get("x", 0.0) or 0.0)
                y = _coerce_float(item.get("y"))
                r = _coerce_float(item.get("R"))
                age = max(0.0, _coerce_float(item.get("age_sec", 0.0) or 0.0))
            except (TypeError, ValueError, OverflowError) as exc:
                _LOG.debug("Skipping malformed GP observation during import: %s", exc)
                continue
            if r < 0:
                continue
            self._xs.append(x)
            self._ys.append(y)
            self._rs.append(r)
            self._ts.append(now - age)
        self._fit_state = None
        self._as_dict_cache = None
        self._fit_inflight = False
        self._fit_reschedule_requested = False
        self._hp_inflight = False
        self._hp_reschedule_requested = False
        if self.n > 0:
            self._fit_cholesky()

    @property
    def sigma2_solo(self) -> float:
        """Estimated solo measurement variance (for interference correction R_t).

        Uses the median of stored R_i values from solo observations (small R).
        Falls back to 10% CV assumption if no observations.
        """
        if not self._rs:
            if self._ys:
                return (0.1 * abs(_coerce_float(np.mean(self._ys)))) ** 2
            return 1.0
        sorted_rs = sorted(self._rs)
        idx = max(0, len(sorted_rs) // 4 - 1)
        return sorted_rs[idx]

    def as_dict(self) -> dict[str, Any]:
        """Serialize for API (backward-compatible fields + GP-specific).

         redesign follow-up () — memoised between
        ``update()`` calls.  The body computes ``predict_grid`` (O(N²)
        Matern kernel over a 30-point x_grid) plus ``confidence(None)``
        (another O(N²) over self._xs); under ops console polling these
        ran N_baselines times per fetch.  The result depends only on
        the training data + hyperparameters, both invariant between
        ``update()`` calls.
        """
        cached = self._as_dict_cache
        if cached is not None:
            return cached
        if self.input_dim == 2:
            result = {
                "mean": round(self.mean, 3),
                "std": round(self.std, 3),
                "n": self.n,
                "confidence": self.confidence(),
                "min": round(min(self._ys), 3) if self._ys else None,
                "max": round(max(self._ys), 3) if self._ys else None,
                "type": "gp",
                "input_dim": 2,
                "kernel": "matern52_product_ard",
                "lengthscales": [round(value, 4) for value in self.lengthscales],
                "sigma2_f": round(self.sigma2_f, 4),
                "beta0": round(self.beta0, 4),
                "beta": [round(value, 6) for value in self.beta],
                "n_total": self._n_total,
                "full_rank_support": self.has_full_rank_support(),
                "observations": {
                    "points": [list(point) for point in self._points],
                    "y": [round(value, 3) for value in self._ys],
                },
                "posterior_curve": None,
            }
            self._as_dict_cache = result
            return result
        posterior_curve: dict[str, Any] | None = None
        if self.n >= 2:
            x_min = min(self._xs)
            x_max = max(self._xs)
            pad = max((x_max - x_min) * 0.1, 1.0)
            x_grid = np.linspace(x_min - pad, x_max + pad, 30).tolist()
            means, variances = self.predict_grid(x_grid)
            stds = [math.sqrt(v) for v in variances]
            posterior_curve = {
                "x": [round(xi, 1) for xi in x_grid],
                "mean": [round(m, 3) for m in means],
                "upper_2sigma": [
                    round(mean + 2 * stds[index], 3) for index, mean in enumerate(means)
                ],
                "lower_2sigma": [
                    round(max(0, mean - 2 * stds[index]), 3)
                    for index, mean in enumerate(means)
                ],
            }

        result = {
            "mean": round(self.mean, 3),
            "std": round(self.std, 3),
            "n": self.n,
            "confidence": self.confidence(),
            "min": round(min(self._ys), 3) if self._ys else None,
            "max": round(max(self._ys), 3) if self._ys else None,
            "type": "gp",
            "kernel": "matern52",
            "lengthscale": round(self.lengthscale, 2),
            "sigma2_f": round(self.sigma2_f, 4),
            "beta0": round(self.beta0, 4),
            "beta1": round(self.beta1, 6),
            "n_total": self._n_total,
            "observations": {
                "x": [round(xi, 1) for xi in self._xs],
                "y": [round(yi, 3) for yi in self._ys],
            },
            "posterior_curve": posterior_curve,
        }
        self._as_dict_cache = result
        return result




@dataclass
class InputScalingModel:
    """Online linear regression for resource scaling with input size.

    Models: resource(x) = intercept + slope * x

    where x is an input size proxy (e.g., sequence length, num_residues).

    Uses Welford-like online updates for both slope and intercept.
    Only fits when we have observations at ≥2 distinct input sizes.
    """

    _sum_x: float = 0.0
    _sum_y: float = 0.0
    _sum_xx: float = 0.0
    _sum_xy: float = 0.0
    _n: int = 0
    _distinct_x: int = 0
    _seen_x: dict[float, int] = field(default_factory=dict)
    _points: list[list[float]] = field(default_factory=list)

    _MAX_POINTS: ClassVar[int] = 200

    def update(self, input_size: float, value: float) -> None:
        self._n += 1
        self._sum_x += input_size
        self._sum_y += value
        self._sum_xx += input_size * input_size
        self._sum_xy += input_size * value
        x_key = round(input_size, 1)
        if x_key not in self._seen_x:
            self._distinct_x += 1
        self._seen_x[x_key] = self._seen_x.get(x_key, 0) + 1
        if len(self._points) < self._MAX_POINTS:
            self._points.append([round(input_size, 1), round(value, 3)])

    @property
    def can_fit(self) -> bool:
        """Need ≥2 distinct input sizes to fit a line."""
        return self._distinct_x >= 2 and self._n >= 3

    @property
    def slope(self) -> float | None:
        if not self.can_fit:
            return None
        denom = self._n * self._sum_xx - self._sum_x * self._sum_x
        if abs(denom) < 1e-12:
            return None
        return (self._n * self._sum_xy - self._sum_x * self._sum_y) / denom

    @property
    def intercept(self) -> float | None:
        s = self.slope
        if s is None:
            return None
        return (self._sum_y - s * self._sum_x) / self._n

    def predict(self, input_size: float) -> float | None:
        s = self.slope
        i = self.intercept
        if s is None or i is None:
            return None
        return i + s * input_size

    def as_dict(self) -> dict[str, Any]:
        return {
            "slope": round(self.slope, 6) if self.slope is not None else None,
            "intercept": round(self.intercept, 3)
            if self.intercept is not None
            else None,
            "n": self._n,
            "distinct_input_sizes": self._distinct_x,
            "can_fit": self.can_fit,
            "points": self._points,
        }




@dataclass
class ConfigBaseline:
    """Per-configuration resource baseline using Gaussian Process regression.

    Tracks VRAM and latency as functions of input_size using independent
    GPEstimate instances.  The GP naturally handles continuous input_size,
    heteroscedastic noise (interference-aware R_i), and provides
    predictions with calibrated uncertainty at any input_size.

    See <docs> .5.3 for mathematical basis.
    """

    config_fingerprint: str
    vram_gp: GPEstimate = field(default_factory=GPEstimate)
    ram_gp: GPEstimate = field(default_factory=GPEstimate)
    latency_gp: GPEstimate = field(default_factory=GPEstimate)
    batch_gps: dict[str, GPEstimate] = field(
        default_factory=lambda: {
            "latency": GPEstimate(input_dim=2),
            "vram": GPEstimate(input_dim=2),
            "ram": GPEstimate(input_dim=2),
        }
    )

    _campaign_obs_count: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    last_updated: float = 0.0

    _predict_vram_cache: dict[float, tuple[float | None, float | None]] = field(
        default_factory=dict
    )
    _predict_ram_cache: dict[float, tuple[float | None, float | None]] = field(
        default_factory=dict
    )
    _predict_latency_cache: dict[float, tuple[float | None, float | None]] = field(
        default_factory=dict
    )
    _vram_confidence_none_cache: float | None = None
    _ram_confidence_none_cache: float | None = None
    _latency_confidence_none_cache: float | None = None
    _resource_prediction_provenance: dict[tuple[str, float, bool], str] = field(
        default_factory=dict
    )
    _resource_fallback_logged: set[tuple[str, float, bool]] = field(default_factory=set)

    temporal_start_gp: GPEstimate = field(default_factory=GPEstimate)
    temporal_duration_gp: GPEstimate = field(default_factory=GPEstimate)
    _temporal_low_vram_n: int = 0
    _temporal_low_vram_mean: float = 0.0
    _temporal_start_n: int = 0
    _temporal_start_mean: float = 0.0
    _temporal_end_n: int = 0
    _temporal_end_mean: float = 0.0


    @property
    def vram_mib(self) -> GPEstimate:
        """GP VRAM estimator (backward compat: .mean, .std, .n, .confidence())."""
        return self.vram_gp

    @property
    def ram_mib(self) -> GPEstimate:
        """GP CPU RAM estimator (task active-memory increment, MiB)."""
        return self.ram_gp

    @property
    def latency_sec(self) -> GPEstimate:
        """GP latency estimator (backward compat: .mean, .std, .n, .confidence())."""
        return self.latency_gp


    def record_vram(
        self,
        vram: float,
        *,
        input_size: float = 0.0,
        campaign_id: str = "",
        R: float | None = None,
    ) -> None:
        x = input_size if input_size > 0 else 0.0
        r = (
            R
            if R is not None
            else max(
                self.vram_gp.sigma2_solo,
                (0.1 * abs(vram)) ** 2 if abs(vram) > 0 else 0.01,
            )
        )
        self.vram_gp.update(x, vram, r)
        if campaign_id:
            self._campaign_obs_count[campaign_id] += 1
        self.last_updated = time.monotonic()
        self._predict_vram_cache.clear()
        self._vram_confidence_none_cache = None
        self._clear_resource_prediction_diagnostics("vram")

    def record_temporal_vram(
        self,
        low_vram: float,
        peak_start_ratio: float,
        peak_end_ratio: float,
        *,
        input_size: float | None = None,
        runtime_sec: float | None = None,
    ) -> None:
        try:
            low = float(low_vram)
            start_ratio = float(peak_start_ratio)
            end_ratio = float(peak_end_ratio)
        except (TypeError, ValueError, OverflowError):
            return
        if (
            not all(math.isfinite(value) for value in (low, start_ratio, end_ratio))
            or low < 0
            or not 0.0 < start_ratio < end_ratio < 1.0
        ):
            return
        self._temporal_low_vram_n += 1
        self._temporal_low_vram_mean += (
            low - self._temporal_low_vram_mean
        ) / self._temporal_low_vram_n
        self.last_updated = time.monotonic()
        if input_size is None or runtime_sec is None:
            self._temporal_start_n += 1
            self._temporal_start_mean += (
                start_ratio - self._temporal_start_mean
            ) / self._temporal_start_n
            self._temporal_end_n += 1
            self._temporal_end_mean += (
                end_ratio - self._temporal_end_mean
            ) / self._temporal_end_n
            return
        try:
            x = float(input_size)
            x = x if x > 0 else 0.0
            wall = float(runtime_sec)
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(x) or not math.isfinite(wall) or wall <= 0:
            return
        start_sec = start_ratio * wall
        duration_sec = (end_ratio - start_ratio) * wall
        for gp, value in (
            (self.temporal_start_gp, start_sec),
            (self.temporal_duration_gp, duration_sec),
        ):
            noise = max(
                gp.sigma2_solo,
                (0.1 * abs(value)) ** 2 if value else 0.01,
            )
            gp.update(x, value, noise)

    def predict_temporal_vram(
        self,
        input_size: float | None = None,
        *,
        z: float = 1.96,
    ) -> dict[str, float] | None:
        low_vram = self._temporal_low_vram_mean
        if self._temporal_low_vram_n < 1 or not math.isfinite(low_vram) or low_vram < 0:
            return None
        if input_size is None:
            if self._temporal_start_n < 1 or self._temporal_end_n < 1:
                return None
            start_ratio = self._temporal_start_mean
            end_ratio = self._temporal_end_mean
            if not 0.0 < start_ratio < end_ratio < 1.0:
                return None
            return {
                "low_vram": low_vram,
                "peak_start_ratio": start_ratio,
                "peak_end_ratio": end_ratio,
            }
        if self.temporal_start_gp.n < 1 or self.temporal_duration_gp.n < 1:
            return None
        try:
            x = float(input_size) if float(input_size) > 0 else 0.0
            start_sec, _ = self.temporal_start_gp.predict(x)
            duration_sec, _ = self.temporal_duration_gp.predict(x)
        except (TypeError, ValueError, OverflowError):
            return None
        peak_end_sec = start_sec + duration_sec
        if not all(
            math.isfinite(value) and value > 0
            for value in (start_sec, duration_sec, peak_end_sec)
        ):
            return None
        return {
            "low_vram": low_vram,
            "peak_start_sec": start_sec,
            "peak_duration_sec": duration_sec,
            "peak_end_sec": peak_end_sec,
        }

    def record_ram(
        self,
        ram: float,
        *,
        input_size: float = 0.0,
        campaign_id: str = "",
        R: float | None = None,
    ) -> None:
        x = input_size if input_size > 0 else 0.0
        r = (
            R
            if R is not None
            else max(
                self.ram_gp.sigma2_solo, (0.1 * abs(ram)) ** 2 if abs(ram) > 0 else 0.01
            )
        )
        self.ram_gp.update(x, ram, r)
        if campaign_id:
            self._campaign_obs_count[campaign_id] += 1
        self.last_updated = time.monotonic()
        self._predict_ram_cache.clear()
        self._ram_confidence_none_cache = None
        self._clear_resource_prediction_diagnostics("ram")

    def record_latency(
        self,
        latency: float,
        *,
        input_size: float = 0.0,
        campaign_id: str = "",
        R: float | None = None,
    ) -> None:
        x = input_size if input_size > 0 else 0.0
        r = (
            R
            if R is not None
            else max(
                self.latency_gp.sigma2_solo,
                (0.1 * abs(latency)) ** 2 if abs(latency) > 0 else 0.01,
            )
        )
        self.latency_gp.update(x, latency, r)
        if campaign_id:
            self._campaign_obs_count[campaign_id] += 1
        self.last_updated = time.monotonic()
        self._predict_latency_cache.clear()
        self._latency_confidence_none_cache = None

    def record_batch_metric(
        self,
        metric: str,
        value: float,
        *,
        input_size: float,
        execution_batch_size: int,
        campaign_id: str = "",
        R: float | None = None,
    ) -> None:
        gp = self.batch_gps.get(metric)
        if gp is None:
            raise ValueError(f"unsupported batch GP metric: {metric}")
        noise = (
            R if R is not None else max(gp.sigma2_solo, (0.1 * abs(value)) ** 2, 0.01)
        )
        gp.update((input_size, execution_batch_size), value, noise)
        if campaign_id:
            self._campaign_obs_count[campaign_id] += 1
        self.last_updated = time.monotonic()

    def predict_batch_grid(
        self,
        metric: str,
        input_size: float,
        execution_batch_sizes: Sequence[int],
    ) -> tuple[list[float], list[float]]:
        gp = self.batch_gps.get(metric)
        if gp is None or gp.n == 0:
            return [], []
        sizes: list[int] = []
        for raw_size in execution_batch_sizes:
            try:
                sizes.append(_coerce_int(raw_size))
            except (TypeError, ValueError) as exc:
                raise ValueError("execution batch sizes must be integers") from exc
        return gp.predict_grid([(input_size, size) for size in sizes])


    def _cached_vram_predict(
        self, input_size: float
    ) -> tuple[float | None, float | None]:
        """fix — returns (mu, var) from cache when available,
        otherwise computes via ``self.vram_gp.predict`` and memoises.
        ``(None, None)`` is the sentinel for ``gp.n == 0``.
        """
        cached = self._predict_vram_cache.get(input_size)
        if cached is not None:
            return cached
        if self.vram_gp.n == 0:
            result: tuple[float | None, float | None] = (None, None)
        else:
            mu, var = self.vram_gp.predict(input_size)
            result = (mu, var)
        self._predict_vram_cache[input_size] = result
        return result

    def _cached_ram_predict(
        self, input_size: float
    ) -> tuple[float | None, float | None]:
        """Mirror of ``_cached_vram_predict`` for CPU RAM."""
        cached = self._predict_ram_cache.get(input_size)
        if cached is not None:
            return cached
        if self.ram_gp.n == 0:
            result: tuple[float | None, float | None] = (None, None)
        else:
            mu, var = self.ram_gp.predict(input_size)
            result = (mu, var)
        self._predict_ram_cache[input_size] = result
        return result

    def _cached_latency_predict(
        self, input_size: float
    ) -> tuple[float | None, float | None]:
        """fix — mirror of ``_cached_vram_predict`` for latency."""
        cached = self._predict_latency_cache.get(input_size)
        if cached is not None:
            return cached
        if self.latency_gp.n == 0:
            result: tuple[float | None, float | None] = (None, None)
        else:
            mu, var = self.latency_gp.predict(input_size)
            result = (mu, var)
        self._predict_latency_cache[input_size] = result
        return result

    @staticmethod
    def _resource_in_support(gp: GPEstimate, input_size: float) -> bool:
        cached = getattr(gp, "_xs_bounds_cache", None)
        n = len(gp._xs)
        if cached is not None and cached[0] == n:
            bounds = cached[1]
        else:
            xs = [
                _coerce_float(value)
                for value in gp._xs
                if isinstance(value, (int, float))
                and math.isfinite(_coerce_float(value))
            ]
            bounds = (min(xs), max(xs)) if xs else None
            gp._xs_bounds_cache = (n, bounds)
        if bounds is None:
            return False
        return bounds[0] <= _coerce_float(input_size) <= bounds[1]

    def _clear_resource_prediction_diagnostics(self, metric: str) -> None:
        self._resource_prediction_provenance = {
            key: value
            for key, value in self._resource_prediction_provenance.items()
            if key[0] != metric
        }
        self._resource_fallback_logged = {
            key for key in self._resource_fallback_logged if key[0] != metric
        }

    def _guard_resource_prediction(
        self,
        metric: str,
        input_size: float,
        mu: float | None,
        var: float | None,
        *,
        z: float | None = None,
    ) -> float | None:
        """Keep scalar GP numerics in-support; abstain to observations outside it."""
        gp = self.vram_gp if metric == "vram" else self.ram_gp
        key = (metric, _coerce_float(input_size), z is not None)
        raw: float | None = None
        if mu is not None and var is not None:
            raw = _coerce_float(mu)
            if z is not None:
                raw += _coerce_float(z) * math.sqrt(max(_coerce_float(var), 0.0))
        if (
            self._resource_in_support(gp, input_size)
            and raw is not None
            and math.isfinite(raw)
            and raw > 0
        ):
            self._resource_prediction_provenance[key] = "gp_in_support"
            return raw

        observed: list[float] = []
        for index, value in enumerate(gp._ys):
            noise = gp._rs[index]
            candidate = _coerce_float(value)
            if z is not None:
                candidate += _coerce_float(z) * math.sqrt(
                    max(_coerce_float(noise), 0.0)
                )
            if math.isfinite(candidate) and candidate > 0:
                observed.append(candidate)
        if not observed:
            self._resource_prediction_provenance[key] = "missing_trusted_observation"
            return None
        provenance = (
            "observed_envelope_out_of_support"
            if not self._resource_in_support(gp, input_size)
            else "observed_envelope_invalid_posterior"
        )
        self._resource_prediction_provenance[key] = provenance
        if key not in self._resource_fallback_logged:
            _LOG.warning(
                "[resource-query] metric=%s config=%s input_size=%.3f "
                "provenance=%s raw=%s fallback=%.3f",
                metric,
                self.config_fingerprint,
                _coerce_float(input_size),
                provenance,
                raw,
                max(observed),
            )
            self._resource_fallback_logged.add(key)
        return max(observed)

    def resource_prediction_provenance(
        self,
        metric: str,
        input_size: float,
        *,
        upper: bool = False,
    ) -> str:
        """Return the most recent scalar resource-query provenance."""
        return self._resource_prediction_provenance.get(
            (str(metric), _coerce_float(input_size), bool(upper)),
            "not_queried",
        )

    def predict_vram(self, input_size: float = 0.0) -> float | None:
        """Predict VRAM, falling back to the observed envelope off-support."""
        mu, var = self._cached_vram_predict(input_size)
        return self._guard_resource_prediction("vram", input_size, mu, var)

    def predict_vram_upper(
        self, input_size: float = 0.0, z: float = 1.96
    ) -> float | None:
        """Predict VRAM upper bound without unsupported GP extrapolation."""
        mu, var = self._cached_vram_predict(input_size)
        return self._guard_resource_prediction("vram", input_size, mu, var, z=z)

    def predict_ram(self, input_size: float = 0.0) -> float | None:
        """Predict CPU RAM, falling back to the observed envelope off-support."""
        mu, var = self._cached_ram_predict(input_size)
        return self._guard_resource_prediction("ram", input_size, mu, var)

    def predict_ram_upper(
        self, input_size: float = 0.0, z: float = 1.96
    ) -> float | None:
        """Predict CPU RAM upper bound without unsupported GP extrapolation."""
        mu, var = self._cached_ram_predict(input_size)
        return self._guard_resource_prediction("ram", input_size, mu, var, z=z)

    def predict_latency(self, input_size: float = 0.0) -> float | None:
        """Predict latency at given input_size via GP posterior mean."""
        mu, _ = self._cached_latency_predict(input_size)
        return mu


    def vram_confidence(self, input_size: float = 0.0) -> float:
        """VRAM confidence at input_size (GP-native: 1 - σ²*/σ²_f).

        Plan fix (A) — routes through ``_cached_vram_predict``
        for the per-input_size path.   redesign follow-up
        () — also memoise the ``input_size <= 0`` fallback
        result; the underlying ``confidence(None)`` calls
        ``predict_grid(self._xs)`` which is O(N²) Matern kernel and was
        the  hang's root hot-path.  The result depends only on the
        training data and is invalidated on the next ``record_vram``.
        """
        if self.vram_gp.n == 0:
            return 0.0
        if input_size > 0 and not self._resource_in_support(self.vram_gp, input_size):
            return 0.0
        if input_size <= 0:
            cached = self._vram_confidence_none_cache
            if cached is not None:
                return cached
            result = self.vram_gp.confidence(None)
            self._vram_confidence_none_cache = result
            return result
        _, var = self._cached_vram_predict(input_size)
        if var is None:
            return 0.0
        raw = 1.0 - var / self.vram_gp.sigma2_f
        return round(max(0.0, min(1.0, raw)), 3)

    def latency_confidence(self, input_size: float = 0.0) -> float:
        """Latency confidence at input_size.

        Plan fix (A) — per-input_size path uses
        ``_cached_latency_predict``.   redesign follow-up — also
        memoise the ``input_size <= 0`` fallback (see ``vram_confidence``
        docstring for the  hang rationale).
        """
        if self.latency_gp.n == 0:
            return 0.0
        if input_size <= 0:
            cached = self._latency_confidence_none_cache
            if cached is not None:
                return cached
            result = self.latency_gp.confidence(None)
            self._latency_confidence_none_cache = result
            return result
        _, var = self._cached_latency_predict(input_size)
        if var is None:
            return 0.0
        raw = 1.0 - var / self.latency_gp.sigma2_f
        return round(max(0.0, min(1.0, raw)), 3)

    def ram_confidence(self, input_size: float = 0.0) -> float:
        """CPU RAM confidence at input_size; mirrors ``vram_confidence``."""
        if self.ram_gp.n == 0:
            return 0.0
        if input_size > 0 and not self._resource_in_support(self.ram_gp, input_size):
            return 0.0
        if input_size <= 0:
            cached = self._ram_confidence_none_cache
            if cached is not None:
                return cached
            result = self.ram_gp.confidence(None)
            self._ram_confidence_none_cache = result
            return result
        _, var = self._cached_ram_predict(input_size)
        if var is None:
            return 0.0
        raw = 1.0 - var / self.ram_gp.sigma2_f
        return round(max(0.0, min(1.0, raw)), 3)

    def is_vram_sufficient(
        self, input_size: float, required_relative_error: float
    ) -> bool:
        """CRLB-backed precision check (fix (A)) — reuses the
        fix ``_cached_vram_predict`` tuple instead of recomputing
        ``GPEstimate.predict`` (which would bypass the cache).
        """
        if self.vram_gp.n == 0 or not self._resource_in_support(
            self.vram_gp, input_size
        ):
            return False
        mu, var = self._cached_vram_predict(input_size)
        if mu is None or var is None:
            return False
        if abs(mu) < 1e-9:
            return 1.96 * math.sqrt(var) < 1.0
        return 1.96 * math.sqrt(var) <= required_relative_error * abs(mu)

    def is_latency_sufficient(
        self, input_size: float, required_relative_error: float
    ) -> bool:
        """CRLB-backed precision check for latency (fix (A))."""
        if self.latency_gp.n == 0:
            return False
        mu, var = self._cached_latency_predict(input_size)
        if mu is None or var is None:
            return False
        if abs(mu) < 1e-9:
            return 1.96 * math.sqrt(var) < 1.0
        return 1.96 * math.sqrt(var) <= required_relative_error * abs(mu)


    def export_state(self) -> dict[str, Any]:
        return {
            "schema": "config_baseline_v3",
            "config_fingerprint": self.config_fingerprint,
            "vram_gp": self.vram_gp.export_state(),
            "ram_gp": self.ram_gp.export_state(),
            "latency_gp": self.latency_gp.export_state(),
            "temporal_start_gp": self.temporal_start_gp.export_state(),
            "temporal_duration_gp": self.temporal_duration_gp.export_state(),
            "batch_gps": {
                name: gp.export_state() for name, gp in self.batch_gps.items()
            },
            "campaign_obs_count": dict(self._campaign_obs_count),
            "temporal_vram": {
                "low_vram_n": self._temporal_low_vram_n,
                "low_vram_mean": self._temporal_low_vram_mean,
                "start_n": self._temporal_start_n,
                "start_mean": self._temporal_start_mean,
                "end_n": self._temporal_end_n,
                "end_mean": self._temporal_end_mean,
            },
            "last_updated_age_sec": (
                max(0.0, time.monotonic() - _coerce_float(self.last_updated))
                if self.last_updated > 0
                else None
            ),
        }

    def import_state(self, data: Mapping[str, Any]) -> None:
        self.config_fingerprint = str(
            data.get("config_fingerprint", self.config_fingerprint)
            or self.config_fingerprint
        )
        self.vram_gp.import_state(_mapping(data.get("vram_gp")))
        self.ram_gp.import_state(_mapping(data.get("ram_gp")))
        self.latency_gp.import_state(_mapping(data.get("latency_gp")))
        self.temporal_start_gp.import_state(_mapping(data.get("temporal_start_gp")))
        self.temporal_duration_gp.import_state(
            _mapping(data.get("temporal_duration_gp"))
        )
        batch_states = data.get("batch_gps") or {}
        if isinstance(batch_states, Mapping):
            for name, gp in self.batch_gps.items():
                gp.import_state(_mapping(batch_states.get(name)))
        counts = data.get("campaign_obs_count") or {}
        self._campaign_obs_count = defaultdict(
            int,
            {
                str(k): _coerce_int(v)
                for k, v in (counts.items() if isinstance(counts, Mapping) else [])
                if str(k)
            },
        )
        temporal = data.get("temporal_vram")
        if isinstance(temporal, Mapping):
            self._temporal_low_vram_n = max(
                0, _coerce_int(temporal.get("low_vram_n", 0))
            )
            self._temporal_low_vram_mean = (
                _optional_float(temporal.get("low_vram_mean")) or 0.0
            )
            self._temporal_start_n = max(0, _coerce_int(temporal.get("start_n", 0)))
            self._temporal_start_mean = (
                _optional_float(temporal.get("start_mean")) or 0.0
            )
            self._temporal_end_n = max(0, _coerce_int(temporal.get("end_n", 0)))
            self._temporal_end_mean = _optional_float(temporal.get("end_mean")) or 0.0
        else:
            self._temporal_low_vram_n = 0
            self._temporal_low_vram_mean = 0.0
            self._temporal_start_n = 0
            self._temporal_start_mean = 0.0
            self._temporal_end_n = 0
            self._temporal_end_mean = 0.0
        age = data.get("last_updated_age_sec")
        try:
            self.last_updated = time.monotonic() - max(0.0, _coerce_float(age))
        except Exception:
            self.last_updated = 0.0
        self._predict_vram_cache.clear()
        self._predict_ram_cache.clear()
        self._predict_latency_cache.clear()
        self._vram_confidence_none_cache = None
        self._ram_confidence_none_cache = None
        self._latency_confidence_none_cache = None
        self._resource_prediction_provenance.clear()
        self._resource_fallback_logged.clear()

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_fingerprint": self.config_fingerprint,
            "vram_mib": self.vram_gp.as_dict(),
            "ram_mib": self.ram_gp.as_dict(),
            "latency_sec": self.latency_gp.as_dict(),
            "batch_gps": {name: gp.as_dict() for name, gp in self.batch_gps.items()},
            "temporal_vram": {
                "low_vram_n": self._temporal_low_vram_n,
                "low_vram_mean": self._temporal_low_vram_mean,
                "peak_start_sec": self.temporal_start_gp.as_dict(),
                "peak_duration_sec": self.temporal_duration_gp.as_dict(),
            },
            "campaigns_observed": len(self._campaign_obs_count),
            "total_observations": sum(self._campaign_obs_count.values()),
        }




@dataclass
class ConfigProfile:
    """Per-(component, config_fingerprint) resource profile with per-GPU sub-baselines.

    The special gpu_id ``"*"`` (``CROSS_GPU``) is **always** populated and acts as the
    cross-GPU pooled fallback.  Per-GPU entries accumulate independently, enabling
    GPU-local predictions where observations are sufficient.

    Observation recording:
    - Every observation is recorded into ``"*"`` (cross-GPU pool).
    - If a ``gpu_id`` is provided, the observation is **also** recorded into the
      GPU-specific baseline, allowing it to diverge from the pool over time.

    Lookup order for predictions:
    1. Exact ``gpu_id`` match  →  GPU-local prediction (most specific).
    2. ``"*"`` (cross-GPU pooled)  →  fallback when no GPU-specific data exists.

    This design ensures a graceful degradation: GPU-specific predictions are used
    when enough data has accumulated; otherwise the system falls back to the
    broader cross-GPU baseline without requiring explicit fallback logic at the
    call site.
    """

    CROSS_GPU: ClassVar[str] = "*"

    config_fingerprint: str
    _gpu_baselines: dict[str, ConfigBaseline] = field(default_factory=dict)

    def get_or_create_gpu(self, gpu_id: str) -> ConfigBaseline:
        if gpu_id not in self._gpu_baselines:
            self._gpu_baselines[gpu_id] = ConfigBaseline(
                config_fingerprint=self.config_fingerprint
            )
        return self._gpu_baselines[gpu_id]

    def resolve(self, gpu_id: str | None = None) -> ConfigBaseline | None:
        """Return the best available baseline for the given gpu_id.

        Falls back to the cross-GPU pooled baseline (``"*"``) when:
        - ``gpu_id`` is None or empty.
        - No observations exist for the specific GPU yet.
        """
        if gpu_id:
            specific = self._gpu_baselines.get(gpu_id)
            if specific is not None:
                return specific
        return self._gpu_baselines.get(self.CROSS_GPU)

    def _check_vram_drift(
        self,
        value: float,
        input_size: float = 0.0,
    ) -> tuple[float, float, float, int] | None:
        """Check whether *value* deviates significantly from the cross-GPU pooled baseline.

        Uses the cross-GPU pool (``"*"``) as the reference because it has the
        most observations and therefore the most reliable prediction.

        Returns ``(observed, predicted, drift_ratio, n_baseline)`` when drift is
        detected, otherwise ``None``.
        """
        cross = self._gpu_baselines.get(self.CROSS_GPU)
        if cross is None or cross.vram_mib.n < _DRIFT_MIN_SAMPLES:
            return None
        predicted = cross.predict_vram(input_size)
        if predicted is None or predicted <= 0:
            return None
        drift_ratio = abs(value - predicted) / predicted
        if drift_ratio >= _DRIFT_THRESHOLD:
            return (value, predicted, drift_ratio, cross.vram_mib.n)
        return None

    def _check_ram_drift(
        self,
        value: float,
        input_size: float = 0.0,
    ) -> tuple[float, float, float, int] | None:
        """Check whether active CPU RAM deviates from the pooled baseline."""
        cross = self._gpu_baselines.get(self.CROSS_GPU)
        if cross is None or cross.ram_mib.n < _DRIFT_MIN_SAMPLES:
            return None
        predicted = cross.predict_ram(input_size)
        if predicted is None or predicted <= 0:
            return None
        drift_ratio = abs(value - predicted) / predicted
        if drift_ratio >= _DRIFT_THRESHOLD:
            return (value, predicted, drift_ratio, cross.ram_mib.n)
        return None

    def _check_latency_drift(
        self,
        value: float,
        input_size: float = 0.0,
    ) -> tuple[float, float, float, int] | None:
        """Check whether *value* deviates significantly from the cross-GPU pooled latency baseline.

        Returns ``(observed, predicted, drift_ratio, n_baseline)`` when drift is
        detected, otherwise ``None``.
        """
        cross = self._gpu_baselines.get(self.CROSS_GPU)
        if cross is None or cross.latency_sec.n < _DRIFT_MIN_SAMPLES:
            return None
        predicted = cross.predict_latency(input_size)
        if predicted is None or predicted <= 0:
            return None
        drift_ratio = abs(value - predicted) / predicted
        if drift_ratio >= _DRIFT_THRESHOLD:
            return (value, predicted, drift_ratio, cross.latency_sec.n)
        return None

    def record_vram(
        self,
        vram: float,
        *,
        input_size: float = 0.0,
        campaign_id: str = "",
        R: float | None = None,
    ) -> None:
        """Record VRAM observation into the cross-GPU pool only.

        VRAM usage is determined by model weights + input size, not by which
        physical GPU is used.  Unlike latency (which differs across GPU models),
        VRAM allocations are identical across GPUs for the same config, so
        per-GPU VRAM baselines add complexity without benefit.
        """
        self.get_or_create_gpu(self.CROSS_GPU).record_vram(
            vram,
            input_size=input_size,
            campaign_id=campaign_id,
            R=R,
        )

    def record_ram(
        self,
        ram: float,
        *,
        input_size: float = 0.0,
        campaign_id: str = "",
        R: float | None = None,
    ) -> None:
        """Record CPU RAM observation into the cross-GPU pool only."""
        self.get_or_create_gpu(self.CROSS_GPU).record_ram(
            ram,
            input_size=input_size,
            campaign_id=campaign_id,
            R=R,
        )

    def record_latency(
        self,
        latency: float,
        *,
        input_size: float = 0.0,
        campaign_id: str = "",
        gpu_id: str | None = None,
        R: float | None = None,
    ) -> None:
        """Record latency observation.  Always populates cross-GPU pool; optionally GPU-specific."""
        self.get_or_create_gpu(self.CROSS_GPU).record_latency(
            latency,
            input_size=input_size,
            campaign_id=campaign_id,
            R=R,
        )
        if gpu_id:
            self.get_or_create_gpu(gpu_id).record_latency(
                latency,
                input_size=input_size,
                campaign_id=campaign_id,
                R=R,
            )

    def record_batch_metric(
        self,
        metric: str,
        value: float,
        *,
        input_size: float,
        execution_batch_size: int,
        campaign_id: str = "",
        gpu_id: str | None = None,
        R: float | None = None,
    ) -> None:
        self.get_or_create_gpu(self.CROSS_GPU).record_batch_metric(
            metric,
            value,
            input_size=input_size,
            execution_batch_size=execution_batch_size,
            campaign_id=campaign_id,
            R=R,
        )
        if metric == "latency" and gpu_id:
            self.get_or_create_gpu(gpu_id).record_batch_metric(
                metric,
                value,
                input_size=input_size,
                execution_batch_size=execution_batch_size,
                campaign_id=campaign_id,
                R=R,
            )

    def get_batch_gp(
        self,
        metric: str,
        gpu_id: str | None = None,
    ) -> GPEstimate:
        if metric == "latency" and gpu_id:
            specific = self._gpu_baselines.get(gpu_id)
            if specific is not None and specific.batch_gps[metric].n > 0:
                return specific.batch_gps[metric]
        return self.get_or_create_gpu(self.CROSS_GPU).batch_gps[metric]

    def predict_vram(
        self, input_size: float = 0.0, gpu_id: str | None = None
    ) -> float | None:
        bl = self.resolve(gpu_id)
        if bl and bl.vram_gp.n > 0:
            return bl.predict_vram(input_size)
        cross = self._gpu_baselines.get(self.CROSS_GPU)
        return cross.predict_vram(input_size) if cross and cross.vram_gp.n > 0 else None

    def predict_vram_upper(
        self, input_size: float = 0.0, gpu_id: str | None = None, z: float = 1.96
    ) -> float | None:
        """Predict VRAM upper bound (μ + z·σ) with GPU-specific → pool fallback."""
        bl = self.resolve(gpu_id)
        if bl and bl.vram_gp.n > 0:
            return bl.predict_vram_upper(input_size, z=z)
        cross = self._gpu_baselines.get(self.CROSS_GPU)
        return (
            cross.predict_vram_upper(input_size, z=z)
            if cross and cross.vram_gp.n > 0
            else None
        )

    def predict_ram(
        self, input_size: float = 0.0, gpu_id: str | None = None
    ) -> float | None:
        bl = self.resolve(gpu_id)
        if bl and bl.ram_gp.n > 0:
            return bl.predict_ram(input_size)
        cross = self._gpu_baselines.get(self.CROSS_GPU)
        return cross.predict_ram(input_size) if cross and cross.ram_gp.n > 0 else None

    def predict_ram_upper(
        self, input_size: float = 0.0, gpu_id: str | None = None, z: float = 1.96
    ) -> float | None:
        """Predict CPU RAM upper bound (μ + z·σ) with GPU-specific → pool fallback."""
        bl = self.resolve(gpu_id)
        if bl and bl.ram_gp.n > 0:
            return bl.predict_ram_upper(input_size, z=z)
        cross = self._gpu_baselines.get(self.CROSS_GPU)
        return (
            cross.predict_ram_upper(input_size, z=z)
            if cross and cross.ram_gp.n > 0
            else None
        )

    def predict_latency(
        self, input_size: float = 0.0, gpu_id: str | None = None
    ) -> float | None:
        bl = self.resolve(gpu_id)
        if bl and bl.latency_gp.n > 0:
            return bl.predict_latency(input_size)
        cross = self._gpu_baselines.get(self.CROSS_GPU)
        return (
            cross.predict_latency(input_size)
            if cross and cross.latency_gp.n > 0
            else None
        )

    def vram_confidence(
        self, gpu_id: str | None = None, input_size: float = 0.0
    ) -> float:
        bl = self.resolve(gpu_id)
        if bl and bl.vram_gp.n > 0:
            return bl.vram_confidence(input_size)
        cross = self._gpu_baselines.get(self.CROSS_GPU)
        return (
            cross.vram_confidence(input_size) if cross and cross.vram_gp.n > 0 else 0.0
        )

    def ram_confidence(
        self, gpu_id: str | None = None, input_size: float = 0.0
    ) -> float:
        bl = self.resolve(gpu_id)
        if bl and bl.ram_gp.n > 0:
            return bl.ram_confidence(input_size)
        cross = self._gpu_baselines.get(self.CROSS_GPU)
        return cross.ram_confidence(input_size) if cross and cross.ram_gp.n > 0 else 0.0

    def latency_confidence(
        self, gpu_id: str | None = None, input_size: float = 0.0
    ) -> float:
        bl = self.resolve(gpu_id)
        if bl and bl.latency_gp.n > 0:
            return bl.latency_confidence(input_size)
        cross = self._gpu_baselines.get(self.CROSS_GPU)
        return (
            cross.latency_confidence(input_size)
            if cross and cross.latency_gp.n > 0
            else 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_fingerprint": self.config_fingerprint,
            "gpu_baselines": {
                gpu_id: bl.as_dict()
                for gpu_id, bl in sorted(self._gpu_baselines.items())
            },
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "schema": "config_profile_v1",
            "config_fingerprint": self.config_fingerprint,
            "gpu_baselines": {
                str(gpu_id): bl.export_state()
                for gpu_id, bl in sorted(self._gpu_baselines.items())
            },
        }

    def import_state(self, data: Mapping[str, Any]) -> None:
        self.config_fingerprint = str(
            data.get("config_fingerprint", self.config_fingerprint)
            or self.config_fingerprint
        )
        self._gpu_baselines = {}
        baselines = data.get("gpu_baselines") or {}
        for gpu_id, payload in (
            baselines.items() if isinstance(baselines, Mapping) else []
        ):
            bl = ConfigBaseline(config_fingerprint=self.config_fingerprint)
            bl.import_state(_mapping(payload))
            self._gpu_baselines[str(gpu_id)] = bl
        if self.CROSS_GPU not in self._gpu_baselines:
            self._gpu_baselines[self.CROSS_GPU] = ConfigBaseline(
                config_fingerprint=self.config_fingerprint
            )




@dataclass
class ComponentResourceProfile:
    """Hierarchical resource profile for a model adapter (component).

    Adapter-level properties (shared across config fingerprints):
    - workload_class: compute_bound / memory_bound / balanced
    - compute characteristics: GPU utilization, arithmetic intensity

    Config-level properties (per config_fingerprint):
    - base VRAM / latency with uncertainty
    - input-size scaling models
    """

    component: str

    _gpu_util_ema: float | None = None
    _execute_us_ema: float | None = None
    _active_memory_ema: float | None = None
    _sample_count: int = 0
    _workload_class: str = "unknown"

    _config_baselines: dict[str, ConfigProfile] = field(default_factory=dict)

    def get_or_create_config(self, config_fingerprint: str) -> ConfigProfile:
        if config_fingerprint not in self._config_baselines:
            self._config_baselines[config_fingerprint] = ConfigProfile(
                config_fingerprint=config_fingerprint,
            )
        return self._config_baselines[config_fingerprint]

    def update_compute_profile(
        self,
        *,
        gpu_util_percent: float | None = None,
        execute_us: float | None = None,
        active_memory_mib: float | None = None,
        is_solo: bool = True,
        concurrent_task_count: int = 1,
    ) -> str:
        """Update adapter-level compute characteristics.

        Solo observations use α=0.30, concurrent use α=0.15 (normalized).
        Returns the updated workload class.
        """
        if not is_solo and concurrent_task_count > 1 and gpu_util_percent is not None:
            gpu_util_percent = gpu_util_percent / concurrent_task_count
        alpha = 0.30 if is_solo else 0.15

        if gpu_util_percent is not None:
            self._gpu_util_ema = (
                gpu_util_percent
                if self._gpu_util_ema is None
                else alpha * gpu_util_percent + (1 - alpha) * self._gpu_util_ema
            )
        if execute_us is not None:
            self._execute_us_ema = (
                execute_us
                if self._execute_us_ema is None
                else alpha * execute_us + (1 - alpha) * self._execute_us_ema
            )
        if active_memory_mib is not None:
            self._active_memory_ema = (
                active_memory_mib
                if self._active_memory_ema is None
                else alpha * active_memory_mib + (1 - alpha) * self._active_memory_ema
            )
        self._sample_count += 1
        self._workload_class = self._classify()
        return self._workload_class

    def _classify(self) -> str:
        if self._gpu_util_ema is not None and self._sample_count >= 2:
            if self._gpu_util_ema >= 65.0:
                return "compute_bound"
            if self._gpu_util_ema <= 35.0:
                return "memory_bound"
            return "balanced"
        if (
            self._execute_us_ema is not None
            and self._active_memory_ema is not None
            and self._active_memory_ema > 0
            and self._sample_count >= 2
        ):
            intensity = self._execute_us_ema / self._active_memory_ema
            if intensity >= 500.0:
                return "compute_bound"
            if intensity <= 100.0:
                return "memory_bound"
            return "balanced"
        return "unknown"

    @property
    def workload_class(self) -> str:
        return self._workload_class

    @property
    def arithmetic_intensity(self) -> float | None:
        """execute_us / active_memory_mib — proxy for compute vs memory bound."""
        if (
            self._execute_us_ema
            and self._active_memory_ema
            and self._active_memory_ema > 0
        ):
            return self._execute_us_ema / self._active_memory_ema
        return None

    def overall_vram_confidence(self) -> float:
        """Aggregate VRAM confidence across all configs (uses cross-GPU pooled baselines)."""
        if not self._config_baselines:
            return 0.0
        confs = [
            cp.vram_confidence()
            for cp in self._config_baselines.values()
            if cp.vram_confidence() > 0
        ]
        return sum(confs) / len(confs) if confs else 0.0

    def overall_ram_confidence(self) -> float:
        """Aggregate CPU RAM confidence across all configs."""
        if not self._config_baselines:
            return 0.0
        confs = [
            cp.ram_confidence()
            for cp in self._config_baselines.values()
            if cp.ram_confidence() > 0
        ]
        return sum(confs) / len(confs) if confs else 0.0

    def overall_latency_confidence(self) -> float:
        if not self._config_baselines:
            return 0.0
        confs = [
            cp.latency_confidence()
            for cp in self._config_baselines.values()
            if cp.latency_confidence() > 0
        ]
        return sum(confs) / len(confs) if confs else 0.0

    def as_dict(self) -> dict[str, Any]:
        intensity = self.arithmetic_intensity
        return {
            "component": self.component,
            "workload_class": self._workload_class,
            "gpu_util_ema": round(self._gpu_util_ema, 1)
            if self._gpu_util_ema is not None
            else None,
            "execute_us_ema": round(self._execute_us_ema, 1)
            if self._execute_us_ema is not None
            else None,
            "active_memory_ema": round(self._active_memory_ema, 1)
            if self._active_memory_ema is not None
            else None,
            "arithmetic_intensity": round(intensity, 1)
            if intensity is not None
            else None,
            "sample_count": self._sample_count,
            "vram_confidence": self.overall_vram_confidence(),
            "ram_confidence": self.overall_ram_confidence(),
            "latency_confidence": self.overall_latency_confidence(),
            "config_baselines": {
                fp: cp.as_dict() for fp, cp in sorted(self._config_baselines.items())
            },
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "schema": "component_resource_profile_v1",
            "component": self.component,
            "gpu_util_ema": self._gpu_util_ema,
            "execute_us_ema": self._execute_us_ema,
            "active_memory_ema": self._active_memory_ema,
            "sample_count": _coerce_int(self._sample_count),
            "workload_class": self._workload_class,
            "config_baselines": {
                str(fp): cp.export_state()
                for fp, cp in sorted(self._config_baselines.items())
            },
        }

    def import_state(self, data: Mapping[str, Any]) -> None:
        self.component = str(data.get("component", self.component) or self.component)
        self._gpu_util_ema = _optional_float(data.get("gpu_util_ema"))
        self._execute_us_ema = _optional_float(data.get("execute_us_ema"))
        self._active_memory_ema = _optional_float(data.get("active_memory_ema"))
        self._sample_count = _coerce_int(data.get("sample_count", 0) or 0)
        self._workload_class = str(data.get("workload_class", "unknown") or "unknown")
        self._config_baselines = {}
        configs = data.get("config_baselines") or {}
        for fp, payload in configs.items() if isinstance(configs, Mapping) else []:
            cfg = ConfigProfile(config_fingerprint=str(fp))
            cfg.import_state(_mapping(payload))
            self._config_baselines[str(fp)] = cfg




class ResourceProfileRegistry:
    """Manages hierarchical resource profiles for all components.

    Key design:
    - Workload class is adapter-level (shared across input sizes / configs)
    - VRAM and latency baselines are per-config
    - Input-size scaling is learned from cross-campaign variation
    - Campaign-aware pooling for intra-campaign similarity

    When a new config_fingerprint appears for a known component, it inherits
    the adapter-level workload class and interference priors. Only the
    config-specific baselines (VRAM, latency) need to be learned from scratch.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ComponentResourceProfile] = {}

    def get_or_create(self, component: str) -> ComponentResourceProfile:
        if component not in self._profiles:
            self._profiles[component] = ComponentResourceProfile(component=component)
        return self._profiles[component]

    def update_compute_profile(
        self,
        component: str,
        *,
        gpu_util_percent: float | None = None,
        execute_us: float | None = None,
        active_memory_mib: float | None = None,
        is_solo: bool = True,
        concurrent_task_count: int = 1,
    ) -> str:
        profile = self.get_or_create(component)
        return profile.update_compute_profile(
            gpu_util_percent=gpu_util_percent,
            execute_us=execute_us,
            active_memory_mib=active_memory_mib,
            is_solo=is_solo,
            concurrent_task_count=concurrent_task_count,
        )

    def classify(self, component: str) -> str:
        profile = self._profiles.get(component)
        return profile.workload_class if profile else "unknown"

    def record_vram(
        self,
        component: str,
        config_fingerprint: str,
        vram_mib: float,
        *,
        input_size: float = 0.0,
        campaign_id: str = "",
        R: float | None = None,
    ) -> tuple[float, float, float, int] | None:
        """Record a VRAM observation and return drift info if a significant deviation is detected.

        VRAM is recorded into the cross-GPU pool (``"*"``) only — VRAM usage is
        config-specific, not GPU-specific (unlike latency which varies by GPU model).

        Drift is checked against the cross-GPU pooled baseline **before** recording the
        new observation (so the check reflects the prior estimate, not the updated one).

        Parameters
        ----------
        R : float, optional
            Kalman measurement noise variance.  If None, defaults to σ²_solo
            (solo-quality observation).

        Returns ``(observed, predicted, drift_ratio, n_baseline)`` when drift is detected,
        otherwise ``None``.
        """
        profile = self.get_or_create(component)
        config = profile.get_or_create_config(config_fingerprint or "__default__")
        drift = config._check_vram_drift(vram_mib, input_size)
        config.record_vram(
            vram_mib, input_size=input_size, campaign_id=campaign_id, R=R
        )
        return drift

    def record_temporal_vram(
        self,
        component: str,
        config_fingerprint: str,
        low_vram: float,
        peak_start_ratio: float,
        peak_end_ratio: float,
        *,
        input_size: float | None = None,
        runtime_sec: float | None = None,
    ) -> None:
        profile = self.get_or_create(component)
        config = profile.get_or_create_config(config_fingerprint or "__default__")
        config.get_or_create_gpu(ConfigProfile.CROSS_GPU).record_temporal_vram(
            low_vram,
            peak_start_ratio,
            peak_end_ratio,
            input_size=input_size,
            runtime_sec=runtime_sec,
        )

    def predict_temporal_vram(
        self,
        component: str,
        config_fingerprint: str,
        *,
        input_size: float | None = None,
        gpu_id: str | None = None,
        z: float = 1.96,
    ) -> dict[str, float] | None:
        profile = self.get_or_create(component)
        config = profile.get_or_create_config(config_fingerprint or "__default__")
        baseline = config.get_or_create_gpu(ConfigProfile.CROSS_GPU)
        return baseline.predict_temporal_vram(input_size=input_size, z=z)

    def record_ram(
        self,
        component: str,
        config_fingerprint: str,
        ram_mib: float,
        *,
        input_size: float = 0.0,
        campaign_id: str = "",
        R: float | None = None,
    ) -> tuple[float, float, float, int] | None:
        """Record active CPU RAM observation and return drift info if detected."""
        profile = self.get_or_create(component)
        config = profile.get_or_create_config(config_fingerprint or "__default__")
        drift = config._check_ram_drift(ram_mib, input_size)
        config.record_ram(ram_mib, input_size=input_size, campaign_id=campaign_id, R=R)
        return drift

    def record_latency(
        self,
        component: str,
        config_fingerprint: str,
        latency_sec: float,
        *,
        input_size: float = 0.0,
        campaign_id: str = "",
        gpu_id: str | None = None,
        R: float | None = None,
    ) -> tuple[float, float, float, int] | None:
        """Record a latency observation and return drift info if a significant deviation is detected.

        Drift is checked against the cross-GPU pooled baseline **before** recording the
        new observation.

        Parameters
        ----------
        R : float, optional
            Kalman measurement noise variance.  If None, defaults to σ²_solo.

        Returns ``(observed, predicted, drift_ratio, n_baseline)`` when drift is detected,
        otherwise ``None``.
        """
        profile = self.get_or_create(component)
        config = profile.get_or_create_config(config_fingerprint or "__default__")
        drift = config._check_latency_drift(latency_sec, input_size)
        config.record_latency(
            latency_sec,
            input_size=input_size,
            campaign_id=campaign_id,
            gpu_id=gpu_id,
            R=R,
        )
        return drift

    def record_batch_metric(
        self,
        component: str,
        config_fingerprint: str,
        metric: str,
        value: float,
        *,
        input_size: float,
        execution_batch_size: int,
        campaign_id: str = "",
        gpu_id: str | None = None,
        R: float | None = None,
    ) -> None:
        profile = self.get_or_create(component)
        config = profile.get_or_create_config(config_fingerprint or "__default__")
        config.record_batch_metric(
            metric,
            value,
            input_size=input_size,
            execution_batch_size=execution_batch_size,
            campaign_id=campaign_id,
            gpu_id=gpu_id,
            R=R,
        )

    def get_batch_gp(
        self,
        component: str,
        config_fingerprint: str,
        metric: str,
        gpu_id: str | None = None,
    ) -> GPEstimate:
        profile = self.get_or_create(component)
        config = profile.get_or_create_config(config_fingerprint or "__default__")
        return config.get_batch_gp(metric, gpu_id)

    def predict_vram(
        self,
        component: str,
        config_fingerprint: str,
        input_size: float = 0.0,
        gpu_id: str | None = None,
    ) -> float | None:
        """Predict VRAM.  Uses GPU-specific baseline if available; falls back to cross-GPU pool."""
        profile = self._profiles.get(component)
        if profile is None:
            return None
        config = profile._config_baselines.get(config_fingerprint or "__default__")
        if config is None:
            return None
        return config.predict_vram(input_size, gpu_id=gpu_id)

    def predict_vram_upper(
        self,
        component: str,
        config_fingerprint: str,
        input_size: float = 0.0,
        gpu_id: str | None = None,
        z: float = 1.96,
    ) -> float | None:
        """Predict VRAM upper bound (μ + z·σ, default 97.5th percentile)."""
        profile = self._profiles.get(component)
        if profile is None:
            return None
        config = profile._config_baselines.get(config_fingerprint or "__default__")
        if config is None:
            return None
        return config.predict_vram_upper(input_size, gpu_id=gpu_id, z=z)

    def predict_ram(
        self,
        component: str,
        config_fingerprint: str,
        input_size: float = 0.0,
        gpu_id: str | None = None,
    ) -> float | None:
        """Predict active CPU RAM. Uses GPU-specific baseline if available; falls back to cross-GPU pool."""
        profile = self._profiles.get(component)
        if profile is None:
            return None
        config = profile._config_baselines.get(config_fingerprint or "__default__")
        if config is None:
            return None
        return config.predict_ram(input_size, gpu_id=gpu_id)

    def predict_ram_upper(
        self,
        component: str,
        config_fingerprint: str,
        input_size: float = 0.0,
        gpu_id: str | None = None,
        z: float = 1.96,
    ) -> float | None:
        """Predict active CPU RAM upper bound (μ + z·σ)."""
        profile = self._profiles.get(component)
        if profile is None:
            return None
        config = profile._config_baselines.get(config_fingerprint or "__default__")
        if config is None:
            return None
        return config.predict_ram_upper(input_size, gpu_id=gpu_id, z=z)

    def predict_latency(
        self,
        component: str,
        config_fingerprint: str,
        input_size: float = 0.0,
        gpu_id: str | None = None,
    ) -> float | None:
        """Predict latency.  Uses GPU-specific baseline if available; falls back to cross-GPU pool."""
        profile = self._profiles.get(component)
        if profile is None:
            return None
        config = profile._config_baselines.get(config_fingerprint or "__default__")
        if config is None:
            return None
        return config.predict_latency(input_size, gpu_id=gpu_id)

    def get_vram_confidence(
        self,
        component: str,
        config_fingerprint: str = "",
        gpu_id: str | None = None,
    ) -> float:
        """Get VRAM prediction confidence for a specific (config, gpu_id) pair."""
        profile = self._profiles.get(component)
        if profile is None:
            return 0.0
        if config_fingerprint:
            config = profile._config_baselines.get(config_fingerprint)
            return config.vram_confidence(gpu_id) if config else 0.0
        return profile.overall_vram_confidence()

    def get_ram_confidence(
        self,
        component: str,
        config_fingerprint: str = "",
        gpu_id: str | None = None,
    ) -> float:
        """Get CPU RAM prediction confidence for a specific (config, gpu_id) pair."""
        profile = self._profiles.get(component)
        if profile is None:
            return 0.0
        if config_fingerprint:
            config = profile._config_baselines.get(config_fingerprint)
            return config.ram_confidence(gpu_id) if config else 0.0
        return profile.overall_ram_confidence()

    def is_vram_sufficient(
        self,
        component: str,
        config_fingerprint: str = "",
        gpu_id: str | None = None,
        required_relative_error: float = 0.10,
    ) -> bool:
        """Check if VRAM estimate meets the required relative precision.

        Uses CRLB-backed criterion: 1.96√P_t / |μ̂| ≤ ε.
        See <docs> .4.
        """
        profile = self._profiles.get(component)
        if profile is None:
            return False
        config = profile._config_baselines.get(config_fingerprint or "__default__")
        if config is None:
            return False
        bl = config.resolve(gpu_id)
        if bl is None:
            return False
        gp = bl.vram_mib
        if hasattr(gp, "is_sufficient") and callable(gp.is_sufficient):
            return gp.is_sufficient(0.0, required_relative_error)
        return False

    def concurrent_vram_decomposition(
        self,
        tasks: list[tuple[str, str]],
        total_vram: float,
        gpu_id: str | None = None,
    ) -> list[float]:
        """Decompose aggregate VRAM across concurrent tasks using learned profiles.

        Uses confidence-weighted proportional attribution:
        - Tasks with confident baselines get proportional shares.
        - Tasks with uncertain baselines get equal-division fallback.

        ``gpu_id`` selects GPU-specific baselines when available; falls back to the
        cross-GPU pooled baseline automatically.

        Returns per-task VRAM estimates (same order as input).
        """
        n = len(tasks)
        if n == 0:
            return []
        if n == 1:
            return [total_vram]

        _DECOMPOSITION_EPSILON = 0.10
        predictions = []
        for comp, cfg in tasks:
            pred = self.predict_vram(comp, cfg, gpu_id=gpu_id)
            sufficient = self.is_vram_sufficient(
                comp,
                cfg,
                gpu_id=gpu_id,
                required_relative_error=_DECOMPOSITION_EPSILON,
            )
            predictions.append((pred, sufficient))

        confident = [
            (i, pred)
            for i, (pred, sufficient) in enumerate(predictions)
            if pred is not None and pred > 0 and sufficient
        ]

        if len(confident) >= n - 1 and len(confident) > 0:
            total_predicted = sum(pred for _, pred in confident)
            if total_predicted <= 0:
                return [total_vram / n] * n

            shares = [0.0] * n
            unknown_indices = []
            for i, (pred, sufficient) in enumerate(predictions):
                if pred is not None and pred > 0 and sufficient:
                    shares[i] = total_vram * (pred / total_predicted)
                else:
                    unknown_indices.append(i)

            allocated = sum(shares)
            if unknown_indices:
                residual = max(0.0, total_vram - allocated)
                per_unknown = residual / len(unknown_indices)
                for i in unknown_indices:
                    shares[i] = per_unknown

            return shares

        return [total_vram / n] * n

    def as_dict(self) -> dict[str, Any]:
        return {
            comp: profile.as_dict() for comp, profile in sorted(self._profiles.items())
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "schema": "resource_profile_registry_v1",
            "profiles": {
                comp: profile.export_state()
                for comp, profile in sorted(self._profiles.items())
            },
        }

    def import_state(self, data: Mapping[str, Any]) -> None:
        self._profiles = {}
        profiles = data.get("profiles") or {}
        for comp, payload in profiles.items() if isinstance(profiles, Mapping) else []:
            profile = ComponentResourceProfile(component=str(comp))
            profile.import_state(_mapping(payload))
            self._profiles[str(comp)] = profile

    @staticmethod
    def extract_input_size(
        workload_features: dict[str, Any],
        component: str = "",
    ) -> float:
        """Extract a scalar input size proxy from workload features.

        Uses the per-component INPUT_SIZE_KEY declared in the adapter's
        workload_features.py (registered at gateway startup).  Falls back to
        the first positive numeric feature if no key is declared.

        Returns 0.0 if no suitable key is found.
        """
        from ..extraction.component_features import get_input_size_key

        preferred = get_input_size_key(component) if component else None
        if preferred:
            val = workload_features.get(preferred)
            if val is not None:
                try:
                    fval = _coerce_float(val)
                    if fval > 0:
                        return fval
                except (TypeError, ValueError):
                    preferred = None

        for key in sorted(workload_features.keys()):
            val = workload_features.get(key)
            if val is not None:
                try:
                    fval = _coerce_float(val)
                    if fval > 0:
                        return fval
                except (TypeError, ValueError):
                    continue
        return 0.0


__all__ = [
    "ComponentResourceProfile",
    "ConfigBaseline",
    "ConfigProfile",
    "GaussianEstimate",
    "InputScalingModel",
    "ResourceProfileRegistry",
]
