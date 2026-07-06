# -*- coding: utf-8 -*-
import os


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import json
import math
import time
import pickle
import traceback
import warnings
from dataclasses import dataclass, asdict
from ctypes import CDLL, POINTER, c_int, c_double
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc, rankdata, wilcoxon, mannwhitneyu, friedmanchisquare

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Global plot settings
GLOBAL_PLOT_PROP = None
_ORIGINAL_PLT_SAVEFIG = None
_SAVEFIG_FONT_PATCHED = False


def _apply_global_figure_font(fig=None, prop=None, fontsize: int = 30):
    global GLOBAL_PLOT_PROP
    if fig is None:
        fig = plt.gcf()
    if prop is None:
        prop = GLOBAL_PLOT_PROP
    if prop is None:
        return

    legend_fontsize = getattr(fig, "_legend_fontsize_override", fontsize)
    tick_fontsize = getattr(fig, "_tick_fontsize_override", fontsize)
    title_fontsize = getattr(fig, "_title_fontsize_override", fontsize)
    label_fontsize = getattr(fig, "_label_fontsize_override", fontsize)


    for txt in fig.findobj(match=matplotlib.text.Text):
        try:
            txt.set_fontproperties(prop)
            txt.set_fontsize(fontsize)
            txt.set_fontweight('bold')
        except Exception:
            pass


    for ax in fig.axes:
        try:
            ax.title.set_fontproperties(prop)
            ax.title.set_fontsize(title_fontsize)
            ax.title.set_fontweight('bold')
            ax.xaxis.label.set_fontproperties(prop)
            ax.xaxis.label.set_fontsize(label_fontsize)
            ax.xaxis.label.set_fontweight('bold')
            ax.yaxis.label.set_fontproperties(prop)
            ax.yaxis.label.set_fontsize(label_fontsize)
            ax.yaxis.label.set_fontweight('bold')
        except Exception:
            pass

        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            try:
                label.set_fontproperties(prop)
                label.set_fontsize(tick_fontsize)
                label.set_fontweight('bold')
            except Exception:
                pass

        legend = ax.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                try:
                    label.set_fontproperties(prop)
                    label.set_fontsize(legend_fontsize)
                    label.set_fontweight('bold')
                except Exception:
                    pass


    for legend in getattr(fig, "legends", []):
        for label in legend.get_texts():
            try:
                label.set_fontproperties(prop)
                label.set_fontsize(legend_fontsize)
                label.set_fontweight('bold')
            except Exception:
                pass

    try:
        fig.canvas.draw_idle()
    except Exception:
        pass

def _patch_savefig_with_global_font():
    global _ORIGINAL_PLT_SAVEFIG, _SAVEFIG_FONT_PATCHED
    if _SAVEFIG_FONT_PATCHED:
        return
    _ORIGINAL_PLT_SAVEFIG = plt.savefig

    def _savefig_with_forced_font(*args, **kwargs):
        _apply_global_figure_font(plt.gcf(), GLOBAL_PLOT_PROP, fontsize=30)
        return _ORIGINAL_PLT_SAVEFIG(*args, **kwargs)

    plt.savefig = _savefig_with_forced_font
    _SAVEFIG_FONT_PATCHED = True


class BoldMathFormatter(ticker.LogFormatterMathtext):
    def __call__(self, x, pos=None):
        if x == 0:
            return "$0$"
        try:
            exponent = int(np.log10(x))
            return r"$\mathbf{10^{%d}}$" % exponent
        except Exception:
            return super().__call__(x, pos)


def convolve_smooth_preserve_ends(curve, window_size: int = 21):
    arr = np.asarray(curve, dtype=np.float64)
    if arr.size == 0:
        return arr
    window_size = int(max(1, window_size))
    if window_size <= 1 or arr.size < 3:
        return arr.copy()
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, arr.size if arr.size % 2 == 1 else arr.size - 1)
    if window_size < 3:
        return arr.copy()
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    pad = window_size // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[0] = arr[0]
    smoothed[-1] = arr[-1]
    return smoothed


@dataclass
class ExperimentConfig:


    SO_PATH: str = "cec17_func.so"


    CEC2017_MODULE_DIR: str = ""


    OUT_ROOT: str = "/public/home/sszzkli/.local/SCLPSO_REVISED12/CEC2017_TRADITIONAL_COMPETITION_NOESCAPE_CEC2014_STYLE_PLOTS"


    FONT_PATH: str = "/public/home/sszzkli/.local/Times New Roman.ttf"
    PLOT_DPI: int = 300


    DIMENSIONS: Tuple[int, ...] = (10, 30, 50, 100)


    EXCLUDE_F2: bool = False
    FUNCTIONS: Tuple[int, ...] = tuple(range(1, 31))


    POP_SIZE: int = 30
    LEGACY_MAX_ITER: int = 1000
    N_RUNS: int = 51


    BUDGET_MODE: str = "cec"
    MAX_FES_FACTOR: int = 10000

    LOWER_BOUND: float = -100.0
    UPPER_BOUND: float = 100.0


    ALGORITHMS: Tuple[str, ...] = (
        "SCLPSO",
        "PSO",
        "DE",
        "CMA-ES",
        "BSO",
        "MFO",
        "SNS",
        "CEALM",
        "HGWODE",
        "HPSOGWO",
    )


    BASELINE_INIT: str = "halton"


    USE_PARALLEL: bool = True
    MAX_PROCESSES: int = min(60, cpu_count() or 1)
    TASKS_PER_CHILD: int = 20
    CHUNKSIZE: int = 1
    SHUFFLE_TASKS: bool = True


    CURVE_POINTS: int = 200


    RESUME: bool = True
    SAVE_EVERY: int = 50


    BASE_SEED: int = 20260604


    TEST_MODE: bool = False


CFG = ExperimentConfig()


def _resolve_cec2017_so_path() -> str:
    candidates = []
    if CFG.SO_PATH:
        candidates.append(CFG.SO_PATH)
    if CFG.CEC2017_MODULE_DIR and CFG.SO_PATH:
        candidates.append(os.path.join(CFG.CEC2017_MODULE_DIR, CFG.SO_PATH))
    if CFG.CEC2017_MODULE_DIR:
        candidates.extend([
            os.path.join(CFG.CEC2017_MODULE_DIR, "cec17_func.so"),
            os.path.join(CFG.CEC2017_MODULE_DIR, "cec17_func.dll"),
        ])
    candidates.extend(["cec17_func.so", "cec17_func.dll"])

    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if os.path.exists(c):
            return os.path.abspath(c)

    raise FileNotFoundError(
        "CEC2017 shared library was not found. Please set CFG.SO_PATH to the full "
        "path of cec17_func.so / cec17_func.dll, or set CFG.CEC2017_MODULE_DIR."
    )


# CEC2017 interface
class CEC17DLLWrapper:

    def __init__(self, dim: int, func_num: int):
        self.dim = int(dim)
        self.func_num = int(func_num)
        self.so_path = _resolve_cec2017_so_path()
        self.so = CDLL(self.so_path)
        self.so.cec17_test_func.argtypes = [
            POINTER(c_double),
            POINTER(c_double),
            c_int,
            c_int,
            c_int,
        ]
        self.so.cec17_test_func.restype = None

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        pop = np.asarray(population, dtype=np.float64)
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        mx, nx = pop.shape
        if nx != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {nx}.")

        x_flat = np.ascontiguousarray(pop, dtype=np.float64).ravel(order="C")
        f_results = np.zeros(mx, dtype=np.float64)
        self.so.cec17_test_func(
            x_flat.ctypes.data_as(POINTER(c_double)),
            f_results.ctypes.data_as(POINTER(c_double)),
            c_int(nx),
            c_int(mx),
            c_int(self.func_num),
        )
        return f_results


def _load_cec2017_wrapper_class():
    so_path = _resolve_cec2017_so_path()
    print(f"CEC2017 shared library: {so_path}", flush=True)
    test_wrapper = CEC17DLLWrapper(dim=10, func_num=1)
    test_x = np.zeros((1, 10), dtype=np.float64)
    test_f = test_wrapper.evaluate_population(test_x)
    if test_f is None or len(test_f) != 1 or not np.isfinite(test_f[0]):
        raise RuntimeError(
            "CEC2017 wrapper sanity check failed: official cec17_test_func "
            "returned an invalid value. Please check the shared library and its "
            "input_data files."
        )
    print(f"CEC2017 wrapper sanity check OK: F1(zeros)={float(test_f[0]):.6e}", flush=True)
    return CEC17DLLWrapper


class BudgetedCEC2017:

    def __init__(self, dim: int, func_num: int, max_fes: int):
        self.wrapper = CEC17DLLWrapper(dim=dim, func_num=func_num)
        self.dim = int(dim)
        self.func_num = int(func_num)
        self.max_fes = int(max_fes)
        self.fes = 0

    def remaining(self) -> int:
        return max(0, self.max_fes - self.fes)

    def can_eval(self, n: int = 1) -> bool:
        return self.fes + int(n) <= self.max_fes

    def eval_pop(self, population: np.ndarray) -> np.ndarray:
        pop = np.asarray(population, dtype=np.float64)
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        if pop.shape[1] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {pop.shape[1]}.")

        n = int(pop.shape[0])
        out = np.full(n, np.inf, dtype=np.float64)
        if n <= 0 or self.fes >= self.max_fes:
            return out

        allowed = min(n, self.max_fes - self.fes)
        if allowed > 0:
            evaluated = self.wrapper.evaluate_population(np.ascontiguousarray(pop[:allowed], dtype=np.float64))
            out[:allowed] = np.asarray(evaluated, dtype=np.float64)
            self.fes += allowed
        return out

    def eval_one(self, x: np.ndarray) -> float:
        return float(self.eval_pop(np.asarray(x, dtype=np.float64).reshape(1, -1))[0])


# Utilities
def max_fes_for_dim(dim: int) -> int:
    if CFG.BUDGET_MODE.lower() == "cec":
        return int(CFG.MAX_FES_FACTOR * dim)
    if CFG.BUDGET_MODE.lower() == "legacy":
        return int(CFG.POP_SIZE * CFG.LEGACY_MAX_ITER)
    raise ValueError("CFG.BUDGET_MODE must be 'legacy' or 'cec'.")


def cec2017_optimum(func_num: int) -> float:

    return 100.0 * float(func_num)


def error_value(best: Any, func_num: int) -> float:
    try:
        val = float(best) - cec2017_optimum(int(func_num))
        if not np.isfinite(val):
            return np.inf
        return max(val, 0.0)
    except Exception:
        return np.inf


def bounds_tuple() -> Tuple[float, float]:
    return float(CFG.LOWER_BOUND), float(CFG.UPPER_BOUND)


def stable_algorithm_index(name: str) -> int:
    if name in CFG.ALGORITHMS:
        return list(CFG.ALGORITHMS).index(name) + 1
    return sum(ord(c) for c in name) % 997 + 1


def make_seed(dim: int, func_num: int, algo: str, run_idx: int) -> int:
    return int(
        CFG.BASE_SEED
        + dim * 1_000_000
        + func_num * 10_000
        + stable_algorithm_index(algo) * 100
        + run_idx
    )


def reflect_bounds(x: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    lb, ub = bounds
    y = np.asarray(x, dtype=np.float64).copy()
    y = np.where(y > ub, 2.0 * ub - y, y)
    y = np.where(y < lb, 2.0 * lb - y, y)
    return np.clip(y, lb, ub)


def clamp_velocity(v: np.ndarray, bounds: Tuple[float, float], vmax_ratio: float = 0.2) -> np.ndarray:
    lb, ub = bounds
    vmax = vmax_ratio * (ub - lb)
    return np.clip(v, -vmax, vmax)


def uniform_init(pop_size: int, dim: int, bounds: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    lb, ub = bounds
    return rng.uniform(lb, ub, size=(pop_size, dim)).astype(np.float64)


def halton_init(pop_size: int, dim: int, bounds: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    lb, ub = bounds
    sampler_seed = int(rng.integers(1, 2**31 - 1))
    sampler = qmc.Halton(d=dim, scramble=True, seed=sampler_seed)
    samples = sampler.random(n=pop_size)
    return qmc.scale(samples, lb, ub).astype(np.float64)


def halton_chaos_init(pop_size: int, dim: int, bounds: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    lb, ub = bounds
    h = halton_init(pop_size, dim, bounds, rng)
    chaos = rng.random((pop_size, dim))
    for _ in range(5):
        chaos = 4.0 * chaos * (1.0 - chaos)
    c = lb + chaos * (ub - lb)
    return np.clip(0.7 * h + 0.3 * c, lb, ub).astype(np.float64)


def baseline_init(pop_size: int, dim: int, bounds: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    if CFG.BASELINE_INIT.lower() == "halton":
        return halton_init(pop_size, dim, bounds, rng)
    return uniform_init(pop_size, dim, bounds, rng)


def population_diversity(pop: np.ndarray, bounds: Tuple[float, float]) -> float:
    lb, ub = bounds
    return float(np.mean(np.std(pop, axis=0)) / (ub - lb + 1e-12))


def levy_step(dim: int, progress: float, rng: np.random.Generator,
              beta_min: float = 1.20, beta_max: float = 1.73) -> np.ndarray:
    beta = beta_min + (beta_max - beta_min) * progress
    numerator = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    denominator = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma = (numerator / denominator) ** (1.0 / beta)
    u = rng.normal(0.0, sigma, dim)
    v = rng.normal(0.0, 1.0, dim)
    scale = 0.3 * (1.0 - progress) + 0.05
    return scale * u / (np.abs(v) ** (1.0 / beta) + 1e-12)


class CurveRecorder:

    def __init__(self, max_fes: int, n_points: int = 200):
        self.max_fes = int(max_fes)
        self.n_points = int(max(10, n_points))
        self.interval = max(1, self.max_fes // self.n_points)
        self.next_fes = 0
        self.fes: List[int] = []
        self.best: List[float] = []
        self.error: List[float] = []
        self.diversity: List[float] = []
        self.state: List[str] = []

    def update(self, fes: int, best_value: float, func_num: int,
               diversity: Optional[float] = None, state: str = ""):
        if fes >= self.next_fes or fes >= self.max_fes:
            self.fes.append(int(fes))
            self.best.append(float(best_value))
            self.error.append(float(error_value(best_value, func_num)))
            self.diversity.append(float(diversity) if diversity is not None else float("nan"))
            self.state.append(str(state))
            self.next_fes += self.interval

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fes": self.fes,
            "best": self.best,
            "error": self.error,
            "diversity": self.diversity,
            "state": self.state,
        }


# SCLPSO settings
SCLPSO_PARAMS: Dict[str, float] = {

    "NINIT_FACTOR": 6.0,
    "MAX_POP": 720,
    "NMIN": 30,


    "CL_REFRESH_GAP": 7,
    "CL_C": 1.49445,
    "GBEST_C_MIN": 0.05,
    "GBEST_C_MAX": 0.85,


    "SHDL_H": 6,
    "SHDL_P": 0.11,
    "SHDL_ARCHIVE_RATE": 2.6,
    "SHDL_RATE_EXPLORATION": 1.00,
    "SHDL_RATE_EXPLOITATION": 0.80,
    "SHDL_RATE_STAGNATION": 0.55,
    "SHDL_RATE_ESCAPE": 0.35,


    "RESTART_THRESHOLD": 14,
    "OBL_RATIO": 0.20,
    "WORST_RESET_RATIO": 0.15,
    "DIVERSITY_LOW": 1e-5,
    "DIVERSITY_MID": 0.08,
    "IMPROVEMENT_EPS": 1e-12,
    "IMPROVEMENT_WINDOW": 5,
}


def _clpso_learning_probability(n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0.05], dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)

    return 0.05 + 0.45 * (np.exp(10.0 * idx / (n - 1.0)) - 1.0) / (np.exp(10.0) - 1.0)


def _sample_lshade_F(memory_f: float, rng: np.random.Generator) -> float:
    for _ in range(50):
        f = memory_f + 0.1 * math.tan(math.pi * (rng.random() - 0.5))
        if f > 0.0:
            return float(min(f, 1.0))
    return 0.5


def _weighted_lehmer_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weights * values))
    if denom <= 1e-300:
        return float(np.mean(values))
    return float(np.sum(weights * values * values) / denom)


def _shade_boundary_correction(mutant: np.ndarray, parent: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    lb, ub = bounds
    y = np.asarray(mutant, dtype=np.float64).copy()
    lower = y < lb
    upper = y > ub
    y[lower] = 0.5 * (parent[lower] + lb)
    y[upper] = 0.5 * (parent[upper] + ub)
    return np.clip(y, lb, ub)


def _diagnose_state(diversity: float, stagnation: int, recent_improvement: float,
                    params: Dict[str, float]) -> str:
    if stagnation >= 2 * int(params["RESTART_THRESHOLD"]):
        return "escape"
    if stagnation >= int(params["RESTART_THRESHOLD"]) or diversity < params["DIVERSITY_LOW"]:
        return "stagnation"
    if recent_improvement <= params["IMPROVEMENT_EPS"] and diversity >= params["DIVERSITY_MID"]:
        return "exploration"
    return "exploitation"


# Optimization algorithms
def sclpso_algorithm(dim: int, func_num: int, pop_size: int, max_fes: int,
                     bounds: Tuple[float, float], seed: int,
                     params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    if params is None:
        params = SCLPSO_PARAMS

    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC2017(dim=dim, func_num=func_num, max_fes=max_fes)
    recorder = CurveRecorder(max_fes=max_fes, n_points=CFG.CURVE_POINTS)
    lb, ub = bounds

    n_min = max(4, int(params["NMIN"]))
    n_min = max(n_min, min(pop_size, int(params["MAX_POP"])))
    n_init = int(round(float(params["NINIT_FACTOR"]) * dim))
    n_init = max(pop_size, n_init, n_min)
    n_init = min(int(params["MAX_POP"]), n_init)
    n_init = min(n_init, max(n_min, max_fes // 10))

    particles = halton_chaos_init(n_init, dim, bounds, rng)
    velocities = np.zeros_like(particles)
    fitness = evaluator.eval_pop(particles)

    pbest_pos = particles.copy()
    pbest_val = fitness.copy()
    g_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[g_idx].copy()
    best_fit = float(pbest_val[g_idx])

    n = particles.shape[0]
    pc = _clpso_learning_probability(n)
    no_improve = np.zeros(n, dtype=np.int64)
    exemplars = np.tile(np.arange(n)[:, None], (1, dim))
    refresh_gap = int(params["CL_REFRESH_GAP"])

    def refresh_particle(i: int):
        nonlocal exemplars, pc, pbest_val
        n_local = pbest_pos.shape[0]
        if n_local <= 1:
            exemplars[i, :] = i
            return
        all_own = True
        for d in range(dim):
            if rng.random() < pc[i] and n_local >= 3:

                a = int(rng.integers(0, n_local))
                while a == i:
                    a = int(rng.integers(0, n_local))
                b = int(rng.integers(0, n_local))
                while b == i or b == a:
                    b = int(rng.integers(0, n_local))
                exemplars[i, d] = a if pbest_val[a] < pbest_val[b] else b
                all_own = False
            else:
                exemplars[i, d] = i
        if all_own and n_local >= 3:
            d = int(rng.integers(0, dim))
            a = int(rng.integers(0, n_local))
            while a == i:
                a = int(rng.integers(0, n_local))
            b = int(rng.integers(0, n_local))
            while b == i or b == a:
                b = int(rng.integers(0, n_local))
            exemplars[i, d] = a if pbest_val[a] < pbest_val[b] else b

    def refresh_all():
        for ii in range(pbest_pos.shape[0]):
            refresh_particle(ii)

    refresh_all()


    h_size = int(params["SHDL_H"])
    m_cr = np.full(h_size, 0.5, dtype=np.float64)
    m_f = np.full(h_size, 0.5, dtype=np.float64)
    memory_pos = 0
    archive = np.empty((0, dim), dtype=np.float64)

    stagnation = 0
    best_history = [best_fit]

    while evaluator.fes < max_fes:
        n = particles.shape[0]
        progress = evaluator.fes / max_fes
        diversity = population_diversity(particles, bounds)

        win = int(params["IMPROVEMENT_WINDOW"])
        if len(best_history) > win:
            prev = best_history[-win - 1]
            recent_improvement = abs(prev - best_fit) / (abs(prev) + 1e-12)
        else:
            recent_improvement = np.inf


        state = "exploitation"


        for ii in range(n):
            if no_improve[ii] >= refresh_gap:
                refresh_particle(ii)
                no_improve[ii] = 0

        dims = np.arange(dim)
        exemplar_pos = np.empty_like(particles)
        for ii in range(n):
            exemplar_pos[ii] = pbest_pos[exemplars[ii], dims]

        w = 0.9 - 0.5 * progress
        c_cl = float(params["CL_C"])
        if state == "exploration":
            c_g = float(params["GBEST_C_MIN"]) * progress
        elif state == "exploitation":
            c_g = float(params["GBEST_C_MIN"]) + (float(params["GBEST_C_MAX"]) - float(params["GBEST_C_MIN"])) * progress
        else:
            c_g = float(params["GBEST_C_MAX"])

        velocities = (
            w * velocities
            + c_cl * rng.random((n, dim)) * (exemplar_pos - particles)
            + c_g * rng.random((n, dim)) * (gbest_pos[None, :] - particles)
        )
        velocities = clamp_velocity(velocities, bounds, 0.2)
        particles = reflect_bounds(particles + velocities, bounds)

        if evaluator.remaining() > 0:
            new_fit = evaluator.eval_pop(particles)
            finite = np.isfinite(new_fit)
            fitness[finite] = new_fit[finite]
            improved = new_fit < pbest_val
            pbest_pos[improved] = particles[improved]
            pbest_val[improved] = new_fit[improved]
            no_improve[improved] = 0
            no_improve[~improved] += 1

            g_idx = int(np.argmin(pbest_val))
            current_best = float(pbest_val[g_idx])
            if current_best < best_fit:
                best_fit = current_best
                gbest_pos = pbest_pos[g_idx].copy()
                stagnation = 0
            else:
                stagnation += 1


        if evaluator.remaining() > 0 and n >= 4:
            if state == "exploration":
                shade_rate = float(params["SHDL_RATE_EXPLORATION"])
            elif state == "exploitation":
                shade_rate = float(params["SHDL_RATE_EXPLOITATION"])
            elif state == "stagnation":
                shade_rate = float(params["SHDL_RATE_STAGNATION"])
            else:
                shade_rate = float(params["SHDL_RATE_ESCAPE"])

            n_trials = max(1, int(round(shade_rate * n)))
            n_trials = min(n_trials, evaluator.remaining(), n)
            trial_indices = rng.choice(n, size=n_trials, replace=False)
            order = np.argsort(pbest_val)
            p_num = max(2, min(n, int(round(float(params["SHDL_P"]) * n))))
            top_indices = order[:p_num]
            union = particles if archive.shape[0] == 0 else np.vstack([particles, archive])
            union_size = union.shape[0]

            trials = np.empty((n_trials, dim), dtype=np.float64)
            cr_values = np.empty(n_trials, dtype=np.float64)
            f_values = np.empty(n_trials, dtype=np.float64)

            for kk, i in enumerate(trial_indices):
                mem_idx = int(rng.integers(0, h_size))
                cr = float(np.clip(rng.normal(m_cr[mem_idx], 0.1), 0.0, 1.0)) if not np.isnan(m_cr[mem_idx]) else 0.0
                f = _sample_lshade_F(float(m_f[mem_idx]), rng)
                cr_values[kk] = cr
                f_values[kk] = f

                pbest_idx = int(rng.choice(top_indices))
                r1 = int(rng.integers(0, n))
                while r1 == i:
                    r1 = int(rng.integers(0, n))
                r2 = int(rng.integers(0, union_size))
                while r2 == i or r2 == r1:
                    r2 = int(rng.integers(0, union_size))

                mutant = particles[i] + f * (pbest_pos[pbest_idx] - particles[i]) + f * (particles[r1] - union[r2])
                mutant = _shade_boundary_correction(mutant, particles[i], bounds)
                cross = rng.random(dim) <= cr
                cross[int(rng.integers(0, dim))] = True
                trials[kk] = np.where(cross, mutant, particles[i])

            trial_fit = evaluator.eval_pop(trials)
            s_cr, s_f, delta_f = [], [], []
            new_archive = []

            for kk, i in enumerate(trial_indices):
                if not np.isfinite(trial_fit[kk]):
                    continue
                if trial_fit[kk] <= fitness[i]:
                    if trial_fit[kk] < fitness[i]:
                        new_archive.append(particles[i].copy())
                        s_cr.append(float(cr_values[kk]))
                        s_f.append(float(f_values[kk]))
                        delta_f.append(float(abs(fitness[i] - trial_fit[kk])))
                    particles[i] = trials[kk]
                    fitness[i] = trial_fit[kk]
                if trial_fit[kk] < pbest_val[i]:
                    pbest_pos[i] = trials[kk]
                    pbest_val[i] = trial_fit[kk]
                    no_improve[i] = 0
                    if trial_fit[kk] < best_fit:
                        best_fit = float(trial_fit[kk])
                        gbest_pos = trials[kk].copy()
                        stagnation = 0

            if new_archive:
                archive = np.vstack([archive, np.asarray(new_archive, dtype=np.float64)])
            archive_max = max(0, int(round(float(params["SHDL_ARCHIVE_RATE"]) * n)))
            if archive.shape[0] > archive_max:
                keep = rng.choice(archive.shape[0], size=archive_max, replace=False)
                archive = archive[keep]

            if s_f:
                s_cr_arr = np.asarray(s_cr, dtype=np.float64)
                s_f_arr = np.asarray(s_f, dtype=np.float64)
                df_arr = np.asarray(delta_f, dtype=np.float64)
                weights = df_arr / np.sum(df_arr) if np.sum(df_arr) > 0 else np.full(len(df_arr), 1.0 / len(df_arr))
                if np.max(s_cr_arr) <= 0.0:
                    m_cr[memory_pos] = np.nan
                else:
                    m_cr[memory_pos] = _weighted_lehmer_mean(s_cr_arr, weights)
                m_f[memory_pos] = _weighted_lehmer_mean(s_f_arr, weights)
                memory_pos = (memory_pos + 1) % h_size


        target_n = int(round(((n_min - n_init) / float(max_fes)) * evaluator.fes + n_init))
        target_n = int(np.clip(target_n, n_min, n_init))
        if target_n < particles.shape[0]:
            keep = np.argsort(pbest_val)[:target_n]
            particles = particles[keep]
            velocities = velocities[keep]
            fitness = fitness[keep]
            pbest_pos = pbest_pos[keep]
            pbest_val = pbest_val[keep]
            no_improve = no_improve[keep]
            pc = _clpso_learning_probability(target_n)
            exemplars = np.tile(np.arange(target_n)[:, None], (1, dim))
            refresh_all()
            archive_max = max(0, int(round(float(params["SHDL_ARCHIVE_RATE"]) * target_n)))
            if archive.shape[0] > archive_max:
                keep_a = rng.choice(archive.shape[0], size=archive_max, replace=False)
                archive = archive[keep_a]

        best_history.append(best_fit)
        recorder.update(evaluator.fes, best_fit, func_num, diversity=diversity, state=f"SCLPSO-{state}")

    return {"best": float(best_fit), "fes": evaluator.fes, "curve": recorder.as_dict()}


def pso_algorithm(dim: int, func_num: int, pop_size: int, max_fes: int,
                  bounds: Tuple[float, float], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC2017(dim, func_num, max_fes)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    x = baseline_init(pop_size, dim, bounds, rng)
    v = np.zeros_like(x)
    fit = evaluator.eval_pop(x)
    pbest = x.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(pbest_fit))
    gbest = pbest[g_idx].copy()
    best = float(pbest_fit[g_idx])

    while evaluator.fes < max_fes:
        progress = evaluator.fes / max_fes
        w = 0.9 - 0.5 * progress
        c1, c2 = 2.0, 2.0
        v = v + c1 * rng.random((pop_size, dim)) * (pbest - x) + c2 * rng.random((pop_size, dim)) * (gbest - x)
        v = clamp_velocity(w * v, bounds, 0.2)
        x = reflect_bounds(x + v, bounds)
        fit = evaluator.eval_pop(x)
        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        g_idx = int(np.argmin(pbest_fit))
        if pbest_fit[g_idx] < best:
            best = float(pbest_fit[g_idx])
            gbest = pbest[g_idx].copy()
        recorder.update(evaluator.fes, best, func_num, population_diversity(x, bounds), "PSO")

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}


def de_algorithm(dim: int, func_num: int, pop_size: int, max_fes: int,
                 bounds: Tuple[float, float], seed: int, F: float = 0.5, CR: float = 0.9) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC2017(dim, func_num, max_fes)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    pop = baseline_init(pop_size, dim, bounds, rng)
    fit = evaluator.eval_pop(pop)
    best = float(np.min(fit))

    while evaluator.fes < max_fes:
        n_eval = min(pop_size, evaluator.remaining())
        trials = np.empty((n_eval, dim), dtype=np.float64)
        indices = np.arange(n_eval)
        for ii, i in enumerate(indices):
            candidates = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = rng.choice(candidates, 3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = reflect_bounds(mutant, bounds)
            cross = rng.random(dim) < CR
            cross[int(rng.integers(0, dim))] = True
            trials[ii] = np.where(cross, mutant, pop[i])
        trial_fit = evaluator.eval_pop(trials)
        for ii, i in enumerate(indices):
            if trial_fit[ii] <= fit[i]:
                pop[i] = trials[ii]
                fit[i] = trial_fit[ii]
        best = min(best, float(np.min(fit)))
        recorder.update(evaluator.fes, best, func_num, population_diversity(pop, bounds), "DE")

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}


def cma_es_algorithm(dim: int, func_num: int, pop_size: int, max_fes: int,
                     bounds: Tuple[float, float], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC2017(dim, func_num, max_fes)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)
    lb, ub = bounds

    lam = int(pop_size)
    mu = max(1, lam // 2)
    weights = np.array([np.log(mu + 0.5) - np.log(i + 1.0) for i in range(mu)], dtype=np.float64)
    weights /= np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)

    mean = rng.uniform(lb, ub, size=dim)
    sigma = 0.3 * (ub - lb)
    var = np.ones(dim, dtype=np.float64)
    ps = np.zeros(dim, dtype=np.float64)
    c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
    c_var = min(0.5, 2.0 / ((dim + 1.3) ** 2 + mu_eff))
    chi_n = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    best = np.inf
    best_x = mean.copy()

    while evaluator.fes < max_fes:
        n_eval = min(lam, evaluator.remaining())
        z = rng.normal(size=(n_eval, dim))
        y = z * np.sqrt(var)[None, :]
        x = np.clip(mean[None, :] + sigma * y, lb, ub)
        fit = evaluator.eval_pop(x)
        finite = np.isfinite(fit)
        if not np.any(finite):
            break
        idx = np.argsort(fit)[:min(mu, n_eval)]
        if fit[idx[0]] < best:
            best = float(fit[idx[0]])
            best_x = x[idx[0]].copy()
        k = len(idx)
        w = weights[:k].copy()
        w /= np.sum(w)
        old_mean = mean.copy()
        mean = np.sum(w[:, None] * x[idx], axis=0)
        z_mean = np.sum(w[:, None] * z[idx], axis=0)
        ps = (1.0 - c_sigma) * ps + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * z_mean
        sigma *= math.exp((np.linalg.norm(ps) / chi_n - 1.0) * c_sigma / d_sigma)
        sigma = float(np.clip(sigma, 1e-12, ub - lb))
        yw = (x[idx] - old_mean[None, :]) / max(sigma, 1e-12)
        var_new = np.sum(w[:, None] * (yw ** 2), axis=0)
        var = (1.0 - c_var) * var + c_var * var_new
        var = np.clip(var, 1e-12, 1e12)
        recorder.update(evaluator.fes, best, func_num, diversity=float(np.mean(np.sqrt(var))), state="CMA-ES")

    if not np.isfinite(best):
        best = evaluator.eval_one(best_x)
    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}


def bso_algorithm(dim: int, func_num: int, pop_size: int, max_fes: int,
                  bounds: Tuple[float, float], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC2017(dim, func_num, max_fes)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)
    lb, ub = bounds

    pop = baseline_init(pop_size, dim, bounds, rng)
    fit = evaluator.eval_pop(pop)
    best = float(np.min(fit))
    best_x = pop[int(np.argmin(fit))].copy()
    n_clusters = max(2, min(5, pop_size // 5))

    while evaluator.fes < max_fes:
        order = np.argsort(fit)
        clusters = np.array_split(order, n_clusters)
        centers = []
        for cl in clusters:
            if len(cl) == 0:
                centers.append(best_x.copy())
            else:

                centers.append(pop[cl[0]].copy())
        centers = np.asarray(centers)

        n_eval = min(pop_size, evaluator.remaining())
        trials = np.empty((n_eval, dim), dtype=np.float64)
        progress = evaluator.fes / max_fes
        noise_scale = 0.25 * (ub - lb) * (1.0 - progress) + 1e-8

        for i in range(n_eval):
            if rng.random() < 0.8:
                c = int(rng.integers(0, n_clusters))
                if rng.random() < 0.4 and len(clusters[c]) > 0:
                    base = pop[int(rng.choice(clusters[c]))]
                else:
                    base = centers[c]
            else:
                c1, c2 = rng.choice(n_clusters, size=2, replace=False)
                a = rng.random()
                base = a * centers[c1] + (1.0 - a) * centers[c2]
            trials[i] = reflect_bounds(base + rng.normal(0.0, noise_scale, size=dim), bounds)

        trial_fit = evaluator.eval_pop(trials)
        for i in range(n_eval):

            worst = int(np.argmax(fit))
            if trial_fit[i] < fit[worst]:
                pop[worst] = trials[i]
                fit[worst] = trial_fit[i]
        idx_best = int(np.argmin(fit))
        if fit[idx_best] < best:
            best = float(fit[idx_best])
            best_x = pop[idx_best].copy()
        recorder.update(evaluator.fes, best, func_num, population_diversity(pop, bounds), "BSO")

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}


def mfo_algorithm(dim: int, func_num: int, pop_size: int, max_fes: int,
                  bounds: Tuple[float, float], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC2017(dim, func_num, max_fes)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)
    lb, ub = bounds

    moths = baseline_init(pop_size, dim, bounds, rng)
    moth_fit = evaluator.eval_pop(moths)
    best = float(np.min(moth_fit))

    flames = moths.copy()
    flame_fit = moth_fit.copy()
    b = 1.0

    while evaluator.fes < max_fes:
        progress = evaluator.fes / max_fes
        combined = np.vstack([flames, moths])
        combined_fit = np.concatenate([flame_fit, moth_fit])
        order = np.argsort(combined_fit)[:pop_size]
        flames = combined[order]
        flame_fit = combined_fit[order]
        if flame_fit[0] < best:
            best = float(flame_fit[0])

        flame_no = int(round(pop_size - progress * (pop_size - 1)))
        flame_no = max(1, min(pop_size, flame_no))
        n_eval = min(pop_size, evaluator.remaining())
        new_moths = moths.copy()
        for i in range(n_eval):
            flame_idx = i if i < flame_no else flame_no - 1
            distance = np.abs(flames[flame_idx] - moths[i])
            t = rng.uniform(-1.0, 1.0, size=dim)
            new_moths[i] = distance * np.exp(b * t) * np.cos(2.0 * np.pi * t) + flames[flame_idx]
        new_moths[:n_eval] = reflect_bounds(new_moths[:n_eval], bounds)
        new_fit = evaluator.eval_pop(new_moths[:n_eval])
        moths[:n_eval] = new_moths[:n_eval]
        moth_fit[:n_eval] = new_fit
        best = min(best, float(np.min(moth_fit)))
        recorder.update(evaluator.fes, best, func_num, population_diversity(moths, bounds), "MFO")

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}


def sns_algorithm(dim: int, func_num: int, pop_size: int, max_fes: int,
                  bounds: Tuple[float, float], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC2017(dim, func_num, max_fes)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)
    lb, ub = bounds

    pop = baseline_init(pop_size, dim, bounds, rng)
    fit = evaluator.eval_pop(pop)
    best_idx = int(np.argmin(fit))
    best = float(fit[best_idx])
    best_x = pop[best_idx].copy()

    while evaluator.fes < max_fes:
        n_eval = min(pop_size, evaluator.remaining())
        trials = np.empty((n_eval, dim), dtype=np.float64)
        progress = evaluator.fes / max_fes
        for i in range(n_eval):
            mood = int(rng.integers(0, 4))
            j = int(rng.integers(0, pop_size))
            while j == i:
                j = int(rng.integers(0, pop_size))
            k = int(rng.integers(0, pop_size))
            while k == i or k == j:
                k = int(rng.integers(0, pop_size))
            if mood == 0:

                x_new = pop[i] + rng.random(dim) * (pop[j] - pop[i]) + rng.random(dim) * (best_x - pop[i])
            elif mood == 1:

                x_new = pop[i] + rng.normal(0, 0.5, size=dim) * (pop[j] - pop[k])
            elif mood == 2:

                mean = np.mean(pop, axis=0)
                x_new = pop[i] + rng.random(dim) * (mean - pop[j]) + rng.random(dim) * (best_x - pop[k])
            else:

                x_new = best_x + rng.normal(0, 0.2 * (ub - lb) * (1.0 - progress) + 1e-8, size=dim)
            trials[i] = reflect_bounds(x_new, bounds)
        trial_fit = evaluator.eval_pop(trials)
        for i in range(n_eval):
            if trial_fit[i] < fit[i]:
                pop[i] = trials[i]
                fit[i] = trial_fit[i]
        idx = int(np.argmin(fit))
        if fit[idx] < best:
            best = float(fit[idx])
            best_x = pop[idx].copy()
        recorder.update(evaluator.fes, best, func_num, population_diversity(pop, bounds), "SNS")

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}


def cealm_algorithm(dim: int, func_num: int, pop_size: int, max_fes: int,
                    bounds: Tuple[float, float], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC2017(dim, func_num, max_fes)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)
    lb, ub = bounds

    lam = int(pop_size)
    mu = max(2, pop_size // 5)
    pop = baseline_init(lam, dim, bounds, rng)
    sigma = np.full((lam, dim), 0.5 * (ub - lb), dtype=np.float64)
    fit = evaluator.eval_pop(pop)
    best = float(np.min(fit))
    tau0 = 1.0 / math.sqrt(2.0 * dim)
    tau = 1.0 / math.sqrt(2.0 * math.sqrt(dim))

    while evaluator.fes < max_fes:
        order = np.argsort(fit)[:mu]
        parents = pop[order]
        parent_sigma = sigma[order]
        n_eval = min(lam, evaluator.remaining())
        offspring = np.empty((n_eval, dim), dtype=np.float64)
        offspring_sigma = np.empty((n_eval, dim), dtype=np.float64)
        for i in range(n_eval):
            p = int(rng.integers(0, mu))
            s = parent_sigma[p].copy()
            s *= np.exp(tau0 * rng.normal() + tau * rng.normal(size=dim))
            s = np.clip(s, 1e-10, ub - lb)
            offspring_sigma[i] = s
            offspring[i] = reflect_bounds(parents[p] + s * rng.normal(size=dim), bounds)
        off_fit = evaluator.eval_pop(offspring)
        combined = np.vstack([pop, offspring])
        combined_sigma = np.vstack([sigma, offspring_sigma])
        combined_fit = np.concatenate([fit, off_fit])
        keep = np.argsort(combined_fit)[:lam]
        pop = combined[keep]
        sigma = combined_sigma[keep]
        fit = combined_fit[keep]
        best = min(best, float(fit[0]))
        recorder.update(evaluator.fes, best, func_num, population_diversity(pop, bounds), "CEALM")

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}


def hgwode_algorithm(dim: int, func_num: int, pop_size: int, max_fes: int,
                     bounds: Tuple[float, float], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC2017(dim, func_num, max_fes)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    pop = baseline_init(pop_size, dim, bounds, rng)
    fit = evaluator.eval_pop(pop)
    best = float(np.min(fit))

    while evaluator.fes < max_fes:
        progress = evaluator.fes / max_fes
        a = 2.0 * (1.0 - progress)
        order = np.argsort(fit)
        alpha, beta, delta = pop[order[0]].copy(), pop[order[1]].copy(), pop[order[2]].copy()
        n_eval = min(pop_size, evaluator.remaining())
        trials = np.empty((n_eval, dim), dtype=np.float64)
        for i in range(n_eval):
            A1 = 2.0 * a * rng.random(dim) - a
            A2 = 2.0 * a * rng.random(dim) - a
            A3 = 2.0 * a * rng.random(dim) - a
            C1 = 2.0 * rng.random(dim)
            C2 = 2.0 * rng.random(dim)
            C3 = 2.0 * rng.random(dim)
            X1 = alpha - A1 * np.abs(C1 * alpha - pop[i])
            X2 = beta - A2 * np.abs(C2 * beta - pop[i])
            X3 = delta - A3 * np.abs(C3 * delta - pop[i])
            gwo = (X1 + X2 + X3) / 3.0

            if rng.random() < 0.5:
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                mutant = pop[r1] + 0.5 * (pop[r2] - pop[r3])
                cr = 0.9
                cross = rng.random(dim) < cr
                cross[int(rng.integers(0, dim))] = True
                trials[i] = np.where(cross, mutant, gwo)
            else:
                trials[i] = gwo
            trials[i] = reflect_bounds(trials[i], bounds)
        trial_fit = evaluator.eval_pop(trials)
        for i in range(n_eval):
            if trial_fit[i] <= fit[i]:
                pop[i] = trials[i]
                fit[i] = trial_fit[i]
        best = min(best, float(np.min(fit)))
        recorder.update(evaluator.fes, best, func_num, population_diversity(pop, bounds), "HGWODE")

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}


def hpsogwo_algorithm(dim: int, func_num: int, pop_size: int, max_fes: int,
                      bounds: Tuple[float, float], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC2017(dim, func_num, max_fes)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    x = baseline_init(pop_size, dim, bounds, rng)
    v = np.zeros_like(x)
    fit = evaluator.eval_pop(x)
    pbest = x.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(pbest_fit))
    gbest = pbest[g_idx].copy()
    best = float(pbest_fit[g_idx])

    while evaluator.fes < max_fes:
        progress = evaluator.fes / max_fes
        a = 2.0 * (1.0 - progress)
        order = np.argsort(pbest_fit)
        alpha, beta, delta = pbest[order[0]].copy(), pbest[order[1]].copy(), pbest[order[2]].copy()
        n_eval = min(pop_size, evaluator.remaining())
        trials = x.copy()
        for i in range(n_eval):
            A1, A2, A3 = 2*a*rng.random(dim)-a, 2*a*rng.random(dim)-a, 2*a*rng.random(dim)-a
            C1, C2, C3 = 2*rng.random(dim), 2*rng.random(dim), 2*rng.random(dim)
            X1 = alpha - A1 * np.abs(C1 * alpha - x[i])
            X2 = beta - A2 * np.abs(C2 * beta - x[i])
            X3 = delta - A3 * np.abs(C3 * delta - x[i])
            gwo_target = (X1 + X2 + X3) / 3.0
            w = 0.9 - 0.5 * progress
            v[i] = w * v[i] + 1.5 * rng.random(dim) * (pbest[i] - x[i]) + 1.5 * rng.random(dim) * (gbest - x[i])
            v[i] = clamp_velocity(v[i], bounds, 0.2)
            pso_pos = x[i] + v[i]
            trials[i] = reflect_bounds(0.5 * pso_pos + 0.5 * gwo_target, bounds)
        new_fit = evaluator.eval_pop(trials[:n_eval])
        x[:n_eval] = trials[:n_eval]
        fit[:n_eval] = new_fit
        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        g_idx = int(np.argmin(pbest_fit))
        if pbest_fit[g_idx] < best:
            best = float(pbest_fit[g_idx])
            gbest = pbest[g_idx].copy()
        recorder.update(evaluator.fes, best, func_num, population_diversity(x, bounds), "HPSOGWO")

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}


ALGORITHM_FUNCS = {
    "SCLPSO": sclpso_algorithm,
    "PSO": pso_algorithm,
    "DE": de_algorithm,
    "CMA-ES": cma_es_algorithm,
    "BSO": bso_algorithm,
    "MFO": mfo_algorithm,
    "SNS": sns_algorithm,
    "CEALM": cealm_algorithm,
    "HGWODE": hgwode_algorithm,
    "HPSOGWO": hpsogwo_algorithm,
}


# Task execution
def run_task(task: Tuple[int, int, str, int]) -> Dict[str, Any]:
    dim, func_num, algo, run_idx = task
    start = time.time()
    seed = make_seed(dim, func_num, algo, run_idx)
    max_fes = max_fes_for_dim(dim)
    bounds = bounds_tuple()
    row: Dict[str, Any] = {
        "dimension": dim,
        "function": func_num,
        "algorithm": algo,
        "run": run_idx,
        "seed": seed,
        "max_fes": max_fes,
        "status": "ok",
        "message": "",
    }
    try:
        func = ALGORITHM_FUNCS[algo]
        result = func(dim, func_num, CFG.POP_SIZE, max_fes, bounds, seed)
        best = float(result["best"])
        curve = result.get("curve", {})
        row.update({
            "best": best,
            "error": error_value(best, func_num),
            "fes": int(result.get("fes", max_fes)),
            "elapsed_sec": time.time() - start,
            "curve_fes_json": json.dumps(curve.get("fes", [])),
            "curve_best_json": json.dumps(curve.get("best", [])),
            "curve_error_json": json.dumps(curve.get("error", [])),
            "curve_state_json": json.dumps(curve.get("state", [])),
        })
    except Exception as exc:
        row.update({
            "status": "error",
            "message": repr(exc) + "\n" + traceback.format_exc(limit=20),
            "best": np.inf,
            "error": np.inf,
            "fes": 0,
            "elapsed_sec": time.time() - start,
            "curve_fes_json": "[]",
            "curve_best_json": "[]",
            "curve_error_json": "[]",
            "curve_state_json": "[]",
        })
    return row


def _current_settings():
    dims = tuple(CFG.DIMENSIONS)
    funcs = tuple(CFG.FUNCTIONS)
    algos = tuple(CFG.ALGORITHMS)
    runs = int(CFG.N_RUNS)
    if CFG.TEST_MODE:
        dims = (dims[0],)
        funcs = tuple(list(funcs)[:2])
        algos = tuple(list(algos)[:3])
        runs = 2
    return dims, funcs, algos, runs


def make_tasks() -> List[Tuple[int, int, str, int]]:
    dims, funcs, algos, runs = _current_settings()
    tasks = [(d, f, a, r) for d in dims for f in funcs for a in algos for r in range(runs)]
    if CFG.SHUFFLE_TASKS:
        rng = np.random.default_rng(CFG.BASE_SEED)
        rng.shuffle(tasks)
    return tasks


def load_existing_results(raw_path: str) -> pd.DataFrame:
    if CFG.RESUME and os.path.exists(raw_path):
        try:
            df = pd.read_csv(raw_path)
            if not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def task_key(row_or_tuple: Any) -> Tuple[int, int, str, int]:
    if isinstance(row_or_tuple, tuple):
        return row_or_tuple
    return (
        int(row_or_tuple["dimension"]),
        int(row_or_tuple["function"]),
        str(row_or_tuple["algorithm"]),
        int(row_or_tuple["run"]),
    )


def save_results(rows: List[Dict[str, Any]], raw_path: str, pickle_path: str):
    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    with open(pickle_path, "wb") as f:
        pickle.dump(rows, f)


# Analysis and plotting
def configure_plot_style() -> font_manager.FontProperties:
    global GLOBAL_PLOT_PROP


    font_path = CFG.FONT_PATH
    if not os.path.exists(font_path):
        raise FileNotFoundError(
            f"Required font file not found: {font_path}. "
            "Please upload Times New Roman.ttf to this path or modify CFG.FONT_PATH."
        )

    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    prop.set_size(30)
    prop.set_weight('bold')
    font_name = prop.get_name()
    GLOBAL_PLOT_PROP = prop

    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.family'] = font_name
    matplotlib.rcParams['font.sans-serif'] = [font_name]
    matplotlib.rcParams['font.serif'] = [font_name]
    matplotlib.rcParams['font.size'] = 30
    matplotlib.rcParams['font.weight'] = 'bold'
    matplotlib.rcParams['axes.titlesize'] = 30
    matplotlib.rcParams['axes.titleweight'] = 'bold'
    matplotlib.rcParams['axes.labelsize'] = 30
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    matplotlib.rcParams['xtick.labelsize'] = 30
    matplotlib.rcParams['ytick.labelsize'] = 30
    matplotlib.rcParams['legend.fontsize'] = 30
    matplotlib.rcParams['figure.titlesize'] = 30
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = font_name
    matplotlib.rcParams['mathtext.it'] = f'{font_name}:italic'
    matplotlib.rcParams['mathtext.bf'] = f'{font_name}:bold'
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True
    matplotlib.rcParams['figure.dpi'] = CFG.PLOT_DPI

    _patch_savefig_with_global_font()
    return prop


def setup_plot_font():
    return configure_plot_style()


def _apply_axis_font(ax, prop, label_size: int = 30, tick_size: int = 30, title_size: int = 30):
    try:
        ax.title.set_fontproperties(prop)
        ax.title.set_fontsize(title_size)
        ax.title.set_fontweight('bold')
        ax.xaxis.label.set_fontproperties(prop)
        ax.xaxis.label.set_fontsize(label_size)
        ax.xaxis.label.set_fontweight('bold')
        ax.yaxis.label.set_fontproperties(prop)
        ax.yaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontweight('bold')
    except Exception:
        pass

    ax.tick_params(axis='both', which='major', labelsize=tick_size, width=2, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3)
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        try:
            lab.set_fontproperties(prop)
            lab.set_fontsize(tick_size)
            lab.set_fontweight('bold')
        except Exception:
            pass


def _safe_plot_name(final: bool, stem: str) -> str:
    return stem if final else stem.replace(".jpg", "_partial.jpg")


def _filter_plot_algorithms(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    allowed = set(CFG.ALGORITHMS)
    if "algorithm" in out.columns:
        out = out[out["algorithm"].isin(allowed)]
    if "competitor" in out.columns:
        out = out[out["competitor"].isin(allowed - {"SCLPSO"})]
    return out


def _algorithm_order_from_df(df: pd.DataFrame) -> List[str]:
    present = set()
    if df is not None and not df.empty:
        if "algorithm" in df.columns:
            present.update(df["algorithm"].dropna().astype(str).unique())
        if "competitor" in df.columns:
            present.update(df["competitor"].dropna().astype(str).unique())
    return [a for a in CFG.ALGORITHMS if a in present]


def _algorithm_color_map(algorithms: Optional[List[str]] = None) -> Dict[str, str]:
    if algorithms is None:
        algorithms = list(CFG.ALGORITHMS)
    palette = [
        '#E41A1C',
        '#377EB8',
        '#FF8C00',
        '#984EA3',
        '#A65628',
        '#F781BF',
        '#666666',
        '#4DAF4A',
        '#984EA3',
        '#FFD700',
        '#1B9E77', '#D95F02', '#7570B3', '#66A61E', '#E6AB02',
    ]
    cmap: Dict[str, str] = {}
    for i, algo in enumerate(algorithms):
        cmap[algo] = palette[i % len(palette)]
    return cmap


def summarize_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = _filter_plot_algorithms(raw_df)
    ok = raw_df[(raw_df["status"] == "ok") & np.isfinite(raw_df["error"])].copy()
    if ok.empty:
        return pd.DataFrame(columns=["dimension", "function", "algorithm", "count", "mean", "std", "min", "median", "max"])
    summary = (
        ok.groupby(["dimension", "function", "algorithm"])["error"]
        .agg(count="count", mean="mean", std="std", min="min", median="median", max="max")
        .reset_index()
    )
    return summary


def ranking_by_dimension(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = _filter_plot_algorithms(summary_df)
    if summary_df.empty:
        return pd.DataFrame(columns=["dimension", "function", "algorithm", "mean_error", "rank"])
    rows = []
    for (dim, func_num), df_case in summary_df.groupby(["dimension", "function"]):
        df_case = df_case.dropna(subset=["mean"]).copy()
        if df_case.empty:
            continue
        ranks = rankdata(df_case["mean"].to_numpy(dtype=float), method="average")
        for (_, row), r in zip(df_case.iterrows(), ranks):
            rows.append({
                "dimension": int(dim),
                "function": int(func_num),
                "algorithm": str(row["algorithm"]),
                "mean_error": float(row["mean"]),
                "rank": float(r),
            })
    return pd.DataFrame(rows)


def win_tie_loss_against_sclpso(summary_df: pd.DataFrame, target: str = "SCLPSO") -> pd.DataFrame:
    summary_df = _filter_plot_algorithms(summary_df)
    rows = []
    if summary_df.empty or target not in set(summary_df.get("algorithm", [])):
        return pd.DataFrame(columns=["dimension", "competitor", "SCLPSO_wins", "ties", "SCLPSO_losses", "wilcoxon_p"])

    for dim, df_dim in summary_df.groupby("dimension"):
        pivot = df_dim.pivot(index="function", columns="algorithm", values="mean")
        if target not in pivot.columns:
            continue
        scl = pivot[target]
        for algo in [a for a in CFG.ALGORITHMS if a != target and a in pivot.columns]:
            pair = pd.concat([scl, pivot[algo]], axis=1, keys=[target, algo]).dropna()
            if pair.empty:
                continue
            scl_vals = pair[target].to_numpy(dtype=float)
            other_vals = pair[algo].to_numpy(dtype=float)
            wins = int(np.sum(scl_vals < other_vals))
            ties = int(np.sum(np.isclose(scl_vals, other_vals)))
            losses = int(np.sum(scl_vals > other_vals))
            p_value = np.nan
            if len(scl_vals) >= 2:
                try:
                    stat = wilcoxon(scl_vals, other_vals, zero_method="wilcox", alternative="two-sided")
                    p_value = float(stat.pvalue)
                except Exception:
                    p_value = np.nan
            rows.append({
                "dimension": int(dim),
                "competitor": algo,
                "SCLPSO_wins": wins,
                "ties": ties,
                "SCLPSO_losses": losses,
                "wilcoxon_p": p_value,
            })
    return pd.DataFrame(rows)


def compute_friedman_average_ranks(summary_df: pd.DataFrame, out_root: str, final: bool = True) -> pd.DataFrame:
    summary_df = _filter_plot_algorithms(summary_df)
    rows = []
    if summary_df.empty:
        return pd.DataFrame(rows)

    rank_df = ranking_by_dimension(summary_df)
    for dim, df_dim in summary_df.groupby("dimension"):
        pivot = df_dim.pivot(index="function", columns="algorithm", values="mean")
        algos_present = [a for a in CFG.ALGORITHMS if a in pivot.columns]
        pivot = pivot[algos_present].dropna()
        stat, p_val = np.nan, np.nan
        if pivot.shape[0] >= 2 and pivot.shape[1] >= 3:
            try:
                stat, p_val = friedmanchisquare(*[pivot[a].to_numpy(dtype=float) for a in algos_present])
            except Exception:
                stat, p_val = np.nan, np.nan
        rd = rank_df[rank_df["dimension"] == dim]
        avg = rd.groupby("algorithm", as_index=False)["rank"].mean()
        for _, row in avg.iterrows():
            rows.append({
                "dimension": int(dim),
                "algorithm": row["algorithm"],
                "avg_rank": float(row["rank"]),
                "friedman_stat": stat,
                "friedman_p": p_val,
            })
    out = pd.DataFrame(rows).sort_values(["dimension", "avg_rank"])
    suffix = "" if final else "_partial"
    out.to_csv(os.path.join(out_root, f"friedman_average_ranks{suffix}.csv"), index=False, encoding="utf-8-sig")
    return out


def save_overall_ranking_table(rank_df: pd.DataFrame, out_root: str, final: bool = True) -> pd.DataFrame:
    rank_df = _filter_plot_algorithms(rank_df)
    if rank_df.empty:
        return pd.DataFrame(columns=["algorithm", "rank", "overall_position"])
    overall = (
        rank_df.groupby("algorithm", as_index=False)["rank"]
        .mean()
        .sort_values("rank", ascending=True)
        .reset_index(drop=True)
    )
    overall["overall_position"] = np.arange(1, len(overall) + 1)
    suffix = "" if final else "_partial"
    overall.to_csv(os.path.join(out_root, f"overall_average_rankings{suffix}.csv"), index=False, encoding="utf-8-sig")
    return overall


def save_mean_std_reports(raw_df: pd.DataFrame, out_root: str, final: bool = True) -> List[str]:
    raw_df = _filter_plot_algorithms(raw_df)
    ok = raw_df[(raw_df["status"] == "ok") & np.isfinite(raw_df["error"])].copy()
    paths: List[str] = []
    if ok.empty:
        return paths

    suffix = "" if final else "_partial"
    for dim, df_dim in ok.groupby("dimension"):
        rows = []
        for func_num, df_func in df_dim.groupby("function"):
            row = {"Function": f"F{int(func_num)}"}
            for algo in CFG.ALGORITHMS:
                vals = df_func[df_func["algorithm"] == algo]["error"].dropna().to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    row[algo] = ""
                else:
                    row[algo] = f"{np.mean(vals):.4e} ± {np.std(vals, ddof=1) if vals.size > 1 else 0.0:.4e}"
            rows.append(row)
        table = pd.DataFrame(rows).sort_values("Function", key=lambda s: s.str.extract(r"F(\d+)")[0].astype(int))
        path = os.path.join(out_root, f"CEC2017_{int(dim)}D_mean_std_table{suffix}.csv")
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)
    return paths


def wilcoxon_rank_sum_vs_sclpso(raw_df: pd.DataFrame, out_root: str, final: bool = True) -> List[str]:
    raw_df = _filter_plot_algorithms(raw_df)
    ok = raw_df[(raw_df["status"] == "ok") & np.isfinite(raw_df["error"])].copy()
    if ok.empty or "SCLPSO" not in set(ok["algorithm"]):
        return []

    rows = []
    for (dim, func_num), df_case in ok.groupby(["dimension", "function"]):
        scl = df_case[df_case["algorithm"] == "SCLPSO"]["error"].dropna().to_numpy(dtype=float)
        scl = scl[np.isfinite(scl)]
        if scl.size < 2:
            continue
        for algo in [a for a in CFG.ALGORITHMS if a != "SCLPSO" and a in set(df_case["algorithm"])] :
            other = df_case[df_case["algorithm"] == algo]["error"].dropna().to_numpy(dtype=float)
            other = other[np.isfinite(other)]
            if other.size < 2:
                continue
            try:
                test = mannwhitneyu(scl, other, alternative="two-sided", method="auto")
                stat = float(test.statistic)
                p_val = float(test.pvalue)
            except Exception:
                stat = np.nan
                p_val = np.nan
            scl_mean = float(np.mean(scl))
            other_mean = float(np.mean(other))
            if np.isfinite(p_val) and p_val < 0.05:
                if scl_mean < other_mean:
                    sign, conclusion = "+", "SCLPSO significantly better"
                elif scl_mean > other_mean:
                    sign, conclusion = "-", "SCLPSO significantly worse"
                else:
                    sign, conclusion = "=", "No practical difference"
            else:
                sign, conclusion = "=", "No significant difference"
            rows.append({
                "dimension": int(dim),
                "function": int(func_num),
                "competitor": algo,
                "test": "Wilcoxon rank-sum / Mann-Whitney U",
                "SCLPSO_mean_error": scl_mean,
                "competitor_mean_error": other_mean,
                "statistic": stat,
                "p_value": p_val,
                "significance_0.05": bool(np.isfinite(p_val) and p_val < 0.05),
                "sign": sign,
                "conclusion": conclusion,
                "n_SCLPSO": int(scl.size),
                "n_competitor": int(other.size),
            })

    if not rows:
        return []
    suffix = "" if final else "_partial"
    test_df = pd.DataFrame(rows).sort_values(["dimension", "function", "competitor"])
    test_path = os.path.join(out_root, f"wilcoxon_rank_sum_vs_SCLPSO{suffix}.csv")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    summary = (
        test_df.groupby("competitor")
        .agg(
            SCLPSO_significantly_better=("sign", lambda s: int(np.sum(np.asarray(s) == "+"))),
            no_significant_difference=("sign", lambda s: int(np.sum(np.asarray(s) == "="))),
            SCLPSO_significantly_worse=("sign", lambda s: int(np.sum(np.asarray(s) == "-"))),
            median_p_value=("p_value", "median"),
            mean_p_value=("p_value", "mean"),
            cases=("p_value", "count"),
        )
        .reset_index()
        .sort_values(["SCLPSO_significantly_better", "SCLPSO_significantly_worse"], ascending=[False, True])
    )
    summary_path = os.path.join(out_root, f"wilcoxon_rank_sum_vs_SCLPSO_summary{suffix}.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return [test_path, summary_path]


def plot_overall_average_rankings(rank_df: pd.DataFrame, out_root: str, final: bool = True) -> Optional[str]:
    rank_df = _filter_plot_algorithms(rank_df)
    overall = save_overall_ranking_table(rank_df, out_root, final=final)
    if overall.empty:
        return None

    prop = configure_plot_style()
    plot_data = overall.sort_values("rank", ascending=True)
    y_pos = np.arange(len(plot_data))
    cmap = _algorithm_color_map(list(plot_data["algorithm"]))

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(y_pos, plot_data["rank"], color=[cmap[a] for a in plot_data["algorithm"]], alpha=0.82)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data["algorithm"], fontproperties=prop, fontsize=30)
    ax.invert_yaxis()
    ax.set_xlabel("Average Rank", fontproperties=prop, fontsize=30, fontweight='bold')
    ax.set_ylabel("Algorithm", fontproperties=prop, fontsize=30, fontweight='bold')
    ax.set_title("Overall Average Ranking on CEC2017 F1-F30", fontproperties=prop, fontsize=30, fontweight='bold')
    ax.grid(True, alpha=0.3, axis="x")
    for bar, rank_value in zip(bars, plot_data["rank"]):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{rank_value:.3f}", ha="left", va="center",
                fontproperties=prop, fontsize=30, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_Overall_Average_Rankings_CEC2017.jpg"))
    plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_dimension_average_rankings(rank_df: pd.DataFrame, out_root: str, final: bool = True) -> Optional[str]:
    rank_df = _filter_plot_algorithms(rank_df)
    if rank_df.empty:
        return None
    prop = configure_plot_style()
    dims = sorted(rank_df["dimension"].unique())
    algos = list(rank_df.groupby("algorithm")["rank"].mean().sort_values().index)
    n_panels = len(dims)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 5.8 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    cmap = _algorithm_color_map(algos)

    for idx, dim in enumerate(dims):
        ax = axes[idx]
        df_dim = rank_df[rank_df["dimension"] == dim].copy()
        df_dim = df_dim.set_index("algorithm").reindex(algos).dropna(subset=["rank"]).reset_index()
        y_pos = np.arange(len(df_dim))
        bars = ax.barh(y_pos, df_dim["rank"], color=[cmap[a] for a in df_dim["algorithm"]], alpha=0.82)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_dim["algorithm"], fontproperties=prop, fontsize=30)
        ax.invert_yaxis()
        ax.set_xlabel("Average Rank", fontproperties=prop, fontsize=30, fontweight='bold')
        ax.set_ylabel("Algorithm", fontproperties=prop, fontsize=30, fontweight='bold')
        ax.set_title(f"CEC2017 {int(dim)}D Average Ranking", fontproperties=prop, fontsize=30, fontweight='bold')
        ax.grid(True, alpha=0.3, axis="x")
        for bar, rank_value in zip(bars, df_dim["rank"]):
            ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                    f"{rank_value:.2f}", ha="left", va="center",
                    fontproperties=prop, fontsize=30, fontweight='bold')
    for j in range(len(dims), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_Dimension_Average_Rankings_CEC2017.jpg"))
    plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
    plt.close()
    return path


def _category_rankings(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = _filter_plot_algorithms(summary_df)
    if summary_df.empty:
        return pd.DataFrame(columns=["category", "algorithm", "rank"])
    groups = {
        "Unimodal (F1-F3)": list(range(1, 4)),
        "Simple multimodal (F4-F10)": list(range(4, 11)),
        "Hybrid (F11-F20)": list(range(11, 21)),
        "Composition (F21-F30)": list(range(21, 31)),
    }
    rows = []
    for category, funcs in groups.items():
        df_cat = summary_df[summary_df["function"].isin(funcs)]
        if df_cat.empty:
            continue
        rank_records = []
        for (dim, func_num), df_func in df_cat.groupby(["dimension", "function"]):
            vals = df_func[["algorithm", "mean"]].dropna()
            if vals.empty:
                continue
            ranks = rankdata(vals["mean"].to_numpy(dtype=float), method="average")
            for (_, row), rank in zip(vals.iterrows(), ranks):
                rank_records.append({"algorithm": row["algorithm"], "rank": float(rank)})
        if not rank_records:
            continue
        tmp = pd.DataFrame(rank_records)
        avg = tmp.groupby("algorithm", as_index=False)["rank"].mean()
        for _, row in avg.iterrows():
            rows.append({"category": category, "algorithm": row["algorithm"], "rank": float(row["rank"])})
    return pd.DataFrame(rows)


def plot_category_rankings(summary_df: pd.DataFrame, out_root: str, final: bool = True) -> Optional[str]:
    cat_df = _category_rankings(summary_df)
    if cat_df.empty:
        return None
    suffix = "" if final else "_partial"
    cat_df.to_csv(os.path.join(out_root, f"category_average_rankings_CEC2017{suffix}.csv"), index=False, encoding="utf-8-sig")
    prop = configure_plot_style()
    categories = list(cat_df["category"].drop_duplicates())
    n_cols = 2
    n_rows = int(math.ceil(len(categories) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.8 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    algos = list(cat_df.groupby("algorithm")["rank"].mean().sort_values().index)
    cmap = _algorithm_color_map(algos)
    for idx, category in enumerate(categories):
        ax = axes[idx]
        df_cat = cat_df[cat_df["category"] == category].set_index("algorithm").reindex(algos).dropna().reset_index()
        y_pos = np.arange(len(df_cat))
        bars = ax.barh(y_pos, df_cat["rank"], color=[cmap[a] for a in df_cat["algorithm"]], alpha=0.82)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_cat["algorithm"], fontproperties=prop, fontsize=30)
        ax.invert_yaxis()
        ax.set_xlabel("Average Rank", fontproperties=prop, fontsize=30, fontweight='bold')
        ax.set_title(category, fontproperties=prop, fontsize=30, fontweight='bold')
        ax.grid(True, alpha=0.3, axis="x")
        for bar, rank_value in zip(bars, df_cat["rank"]):
            ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                    f"{rank_value:.2f}", ha="left", va="center", fontproperties=prop, fontsize=30, fontweight='bold')
    for j in range(len(categories), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_Category_Average_Rankings_CEC2017.jpg"))
    plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_win_tie_loss(wtl_df: pd.DataFrame, out_root: str, final: bool = True) -> Optional[str]:
    wtl_df = _filter_plot_algorithms(wtl_df)
    if wtl_df.empty:
        return None
    prop = configure_plot_style()
    plot_df = (
        wtl_df.groupby("competitor", as_index=False)[["SCLPSO_wins", "ties", "SCLPSO_losses"]]
        .sum()
        .sort_values(["SCLPSO_wins", "SCLPSO_losses"], ascending=[False, True])
    )
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(12, max(5, 0.55 * len(plot_df))))
    ax.barh(y, plot_df["SCLPSO_wins"], label="SCLPSO wins", alpha=0.85)
    ax.barh(y, plot_df["ties"], left=plot_df["SCLPSO_wins"], label="Ties", alpha=0.85)
    ax.barh(y, plot_df["SCLPSO_losses"], left=plot_df["SCLPSO_wins"] + plot_df["ties"], label="SCLPSO losses", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["competitor"], fontproperties=prop, fontsize=30)
    ax.invert_yaxis()
    ax.set_xlabel("Number of function cases", fontproperties=prop, fontsize=30, fontweight='bold')
    ax.set_title("Win/Tie/Loss Counts of SCLPSO vs Traditional Algorithms on CEC2017", fontproperties=prop, fontsize=30, fontweight='bold')
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(prop=prop, fontsize=30)
    plt.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_Win_Tie_Loss_CEC2017.jpg"))
    plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
    plt.close()
    return path


def _parse_curve_json(s: Any) -> List[float]:
    if isinstance(s, str):
        try:
            return list(json.loads(s))
        except Exception:
            return []
    if isinstance(s, (list, tuple, np.ndarray)):
        return list(s)
    return []


def _curve_error_arrays(df_case: pd.DataFrame, algo: str, max_fes: int, n_points: int = 200) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    ga = df_case[df_case["algorithm"] == algo]
    if ga.empty:
        return None

    common_fes = np.linspace(0, int(max_fes), 10000)
    x_plot = common_fes
    curves = []
    for _, row in ga.iterrows():
        fes = np.asarray(_parse_curve_json(row.get("curve_fes_json", "[]")), dtype=float)
        err = np.asarray(_parse_curve_json(row.get("curve_error_json", "[]")), dtype=float)
        if len(fes) < 2 or len(err) < 2:
            continue
        n = min(len(fes), len(err))
        fes, err = fes[:n], err[:n]
        mask = np.isfinite(fes) & np.isfinite(err)
        fes, err = fes[mask], err[mask]
        if len(fes) < 2:
            continue
        order = np.argsort(fes)
        fes, err = fes[order], np.maximum(err[order], 0.0)
        uniq, idx = np.unique(fes, return_index=True)
        fes, err = uniq, err[idx]
        if len(fes) < 2:
            continue
        fill_value = np.nanmax(err[np.isfinite(err)]) if np.any(np.isfinite(err)) else 1.0
        err = np.nan_to_num(err, nan=fill_value, posinf=fill_value, neginf=0.0)
        interp = np.interp(common_fes, fes, err)
        window_size = max(3, min(21, int(len(interp) * 0.01)))
        interp = convolve_smooth_preserve_ends(interp, window_size)
        curves.append(interp)
    if not curves:
        return None
    y = np.median(np.vstack(curves), axis=0)
    return x_plot, np.maximum(y, 1e-30)


def generate_algorithm_legend(out_root: str, final: bool = True) -> Optional[str]:
    if not final:
        return None
    prop = configure_plot_style()
    algos = list(CFG.ALGORITHMS)
    cmap = _algorithm_color_map(algos)
    fig, ax = plt.subplots(figsize=(8, 4))
    handles = []
    for algo in algos:
        line, = ax.plot([], [], label=algo, color=cmap.get(algo, '#377EB8'), linestyle='-', linewidth=2.5)
        handles.append(line)
    leg = ax.legend(handles=handles, loc='center', ncol=5, prop=prop, fontsize=30, framealpha=0.8)
    for txt in leg.get_texts():
        txt.set_fontproperties(prop)
        txt.set_fontsize(30)
        txt.set_fontweight('bold')
    ax.axis('off')
    fig.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, 'SCLPSO_algorithm_legend.jpg'))
    fig.savefig(path, dpi=700, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_convergence_curves(curve_df: pd.DataFrame, out_root: str, final: bool = True) -> Optional[str]:
    curve_df = _filter_plot_algorithms(curve_df)
    ok = curve_df[(curve_df["status"] == "ok") & np.isfinite(curve_df["error"])].copy()
    if ok.empty:
        return None

    prop = configure_plot_style()
    cmap = _algorithm_color_map(list(CFG.ALGORITHMS))

    preferred = [(10, 1), (30, 10), (50, 23), (100, 30)]
    available = set((int(d), int(f)) for d, f in ok[["dimension", "function"]].drop_duplicates().itertuples(index=False, name=None))
    selected_cases = [case for case in preferred if case in available]
    if len(selected_cases) < 4:
        for case in sorted(available):
            if case not in selected_cases:
                selected_cases.append(case)
            if len(selected_cases) >= 4:
                break
    if not selected_cases:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    epsilon = 1e-30

    for ax_idx, (dim, func_num) in enumerate(selected_cases[:4]):
        ax = axes[ax_idx]
        df_case = ok[(ok["dimension"] == dim) & (ok["function"] == func_num)]
        max_fes = max_fes_for_dim(int(dim))
        all_valid_points: List[float] = []
        plot_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for algo in CFG.ALGORITHMS:
            arr = _curve_error_arrays(df_case, algo, max_fes, n_points=240)
            if arr is None:
                continue
            x, y = arr
            y = np.maximum(y, epsilon)
            plot_data[algo] = (x, y)
            valid = y > epsilon * 1.1
            if np.any(valid):
                all_valid_points.extend(y[valid].tolist())

        if not plot_data:
            ax.axis('off')
            continue

        if all_valid_points:
            y_min = max(min(all_valid_points), epsilon)
            if y_min < 1e-20:
                y_min = epsilon
            initial_values = [yy[0] for _, yy in plot_data.values() if len(yy) > 0]
            y_max = max(initial_values) if initial_values else max(all_valid_points)
            ax.set_ylim(y_min * 0.1, y_max * 2.0)

        for algo in CFG.ALGORITHMS:
            if algo not in plot_data:
                continue
            x, y = plot_data[algo]
            ax.semilogy(
                x,
                y,
                label=algo,
                color=cmap.get(algo),
                linestyle='-',
                linewidth=2.5,
                alpha=1.0,
            )

        ax.yaxis.set_major_formatter(BoldMathFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.set_xlabel('Function Evaluations (FEs)', fontproperties=prop, fontsize=30, fontweight='bold')
        ax.set_ylabel('Mean Error (log scale)', fontproperties=prop, fontsize=30, fontweight='bold')
        ax.grid(True, which='major', alpha=0.4, linestyle='--', linewidth=1)
        ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
        _apply_axis_font(ax, prop, label_size=30, tick_size=30, title_size=30)

    for j in range(len(selected_cases), len(axes)):
        axes[j].axis('off')

    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.08, wspace=0.28, hspace=0.34)
    path = os.path.join(out_root, _safe_plot_name(final, 'SCLPSO_Selected_Convergence_Curves_CEC2017.jpg'))
    fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return path

def plot_convergence_curves_by_function(curve_df: pd.DataFrame, out_root: str, final: bool = True) -> List[str]:
    curve_df = _filter_plot_algorithms(curve_df)
    ok = curve_df[(curve_df["status"] == "ok") & np.isfinite(curve_df["error"])].copy()
    if ok.empty:
        return []

    prop = configure_plot_style()
    fig_dir = os.path.join(out_root, 'convergence_curves_by_function')
    os.makedirs(fig_dir, exist_ok=True)
    paths: List[str] = []
    cmap = _algorithm_color_map(list(CFG.ALGORITHMS))
    epsilon = 1e-30

    def plot_one_case(dim: int, func_num: int, plot_type: str = 'normal') -> Optional[str]:
        df_case = ok[(ok['dimension'] == dim) & (ok['function'] == func_num)]
        max_fes = max_fes_for_dim(int(dim))
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        all_valid_points: List[float] = []
        final_values: List[float] = []

        for algo in CFG.ALGORITHMS:
            arr = _curve_error_arrays(df_case, algo, max_fes, n_points=240)
            if arr is None:
                continue
            x, y = arr
            y = np.maximum(y, epsilon)
            plot_data[algo] = (x, y)
            valid = y > epsilon * 1.1
            if np.any(valid):
                all_valid_points.extend(y[valid].tolist())
            if len(y) > 0 and np.isfinite(y[-1]):
                final_values.append(float(y[-1]))

        if not plot_data or not all_valid_points:
            plt.close(fig)
            return None

        y_min = max(min(all_valid_points), epsilon)
        if y_min < 1e-20:
            y_min = epsilon
        initial_values = [yy[0] for _, yy in plot_data.values() if len(yy) > 0]
        y_max = max(initial_values) if initial_values else max(all_valid_points)
        ax.set_ylim(y_min * 0.1, y_max * 2.0)

        axins = None
        if plot_type == 'zoomed':
            axins = inset_axes(
                ax,
                width='40%',
                height='30%',
                loc='center',
                bbox_to_anchor=(0.15, 0.15, 0.7, 0.7),
                bbox_transform=ax.transAxes,
            )

        for algo in CFG.ALGORITHMS:
            if algo not in plot_data:
                continue
            x, y = plot_data[algo]
            color = cmap.get(algo)
            ax.semilogy(x, y, color=color, linestyle='-', linewidth=2.5, label=algo)
            if axins is not None:
                axins.semilogy(x, y, color=color, linestyle='-', linewidth=2.5)

        if plot_type == 'zoomed' and axins is not None:
            x_ref = next(iter(plot_data.values()))[0]
            x_start = x_ref[int(len(x_ref) * 0.8)]
            x_end = x_ref[-1]
            axins.set_xlim(x_start, x_end)
            sorted_finals = sorted([v for v in final_values if np.isfinite(v) and v > 0])
            if sorted_finals:
                y_zoom_min = max(sorted_finals[0] * 0.5, epsilon)
                cutoff_idx = min(len(sorted_finals) - 1, 3)
                y_zoom_max = sorted_finals[cutoff_idx] * 5.0
                if y_zoom_max <= y_zoom_min:
                    y_zoom_max = y_zoom_min * 10.0
                axins.set_ylim(y_zoom_min, y_zoom_max)
            axins.set_xticklabels([])
            axins.set_yticklabels([])
            axins.minorticks_on()
            mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.5', linestyle='--', linewidth=0.8)

        ax.yaxis.set_major_formatter(BoldMathFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.set_xlabel('Function Evaluations (FEs)', fontproperties=prop, fontsize=30, fontweight='bold')
        ax.set_ylabel('Mean Error (log scale)', fontproperties=prop, fontsize=30, fontweight='bold')
        ax.grid(True, which='major', alpha=0.4, linestyle='--', linewidth=1)
        ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
        _apply_axis_font(ax, prop, label_size=30, tick_size=30, title_size=30)

        suffix = 'zoomed' if plot_type == 'zoomed' else 'log_error'
        path = os.path.join(fig_dir, _safe_plot_name(final, f'CEC2017_{int(dim)}D_F{int(func_num):02d}_{suffix}.jpg'))
        fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        return path

    for (dim, func_num), _ in ok.groupby(['dimension', 'function']):
        for plot_type in ('normal', 'zoomed'):
            path = plot_one_case(int(dim), int(func_num), plot_type=plot_type)
            if path:
                paths.append(path)
    return paths

def plot_boxplots_by_function(raw_df: pd.DataFrame, out_root: str, final: bool = True) -> List[str]:
    raw_df = _filter_plot_algorithms(raw_df)
    ok = raw_df[(raw_df["status"] == "ok") & np.isfinite(raw_df["best"])].copy()
    if ok.empty:
        return []
    prop = configure_plot_style()
    fig_dir = os.path.join(out_root, 'boxplots_by_function')
    os.makedirs(fig_dir, exist_ok=True)
    paths: List[str] = []
    cmap = _algorithm_color_map(list(CFG.ALGORITHMS))

    for (dim, func_num), df_case in ok.groupby(['dimension', 'function']):
        data, labels, colors = [], [], []
        for algo in CFG.ALGORITHMS:
            vals = df_case[df_case['algorithm'] == algo]['best'].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size >= 2:
                data.append(vals)
                labels.append(algo)
                colors.append(cmap.get(algo, '#377EB8'))
        if not data:
            continue

        fig, ax = plt.subplots(figsize=(12, 8))
        all_values = np.concatenate(data) if data else np.array([])
        if all_values.size > 0:
            min_val = float(np.min(all_values))
            max_val = float(np.max(all_values))
            value_range = max_val / min_val if min_val > 0 else 1.0
            use_log_scale = value_range > 1e4
        else:
            use_log_scale = False

        bp = ax.boxplot(
            data,
            patch_artist=True,
            vert=True,
            showfliers=True,
            showmeans=True,
            meanline=True,
            meanprops={'color': 'red', 'linewidth': 1.5},
        )
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontproperties=prop, fontweight='bold', fontsize=30)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for element in ['whiskers', 'caps', 'medians', 'means']:
            if element in bp:
                for item in bp[element]:
                    try:
                        item.set_linewidth(1.5)
                    except Exception:
                        pass

        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Objective Value (log scale)', fontproperties=prop, fontweight='bold', fontsize=30)
            ax.yaxis.set_major_formatter(BoldMathFormatter())
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        else:
            ax.set_ylabel('Objective Value', fontproperties=prop, fontweight='bold', fontsize=30)

        ax.grid(True, axis='y', alpha=0.5)
        _apply_axis_font(ax, prop, label_size=30, tick_size=30, title_size=30)
        fig.tight_layout()
        path = os.path.join(fig_dir, _safe_plot_name(final, f'CEC2017_{int(dim)}D_F{int(func_num):02d}_boxplot.jpg'))
        fig.savefig(path, dpi=700, bbox_inches='tight')
        plt.close(fig)
        paths.append(path)
    return paths


def plot_average_rank_radar(summary_df: pd.DataFrame, out_root: str, final: bool = True) -> List[str]:
    summary_df = _filter_plot_algorithms(summary_df)
    if summary_df.empty:
        return []
    prop = configure_plot_style()
    paths: List[str] = []
    cmap = _algorithm_color_map(list(CFG.ALGORITHMS))
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]

    for dim, df_dim in summary_df.groupby("dimension"):
        pivot = df_dim.pivot(index="function", columns="algorithm", values="mean")
        funcs = sorted([int(f) for f in pivot.index])
        algos = [a for a in CFG.ALGORITHMS if a in pivot.columns]
        if len(funcs) < 3 or not algos:
            continue
        pivot = pivot.loc[funcs, algos]
        normalized = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
        for func_num in pivot.index:
            values = pivot.loc[func_num].astype(float)
            finite = np.isfinite(values.to_numpy(dtype=float))
            if not finite.any():
                normalized.loc[func_num] = 0.0
                continue
            max_val = values[finite].max()
            min_val = values[finite].min()
            if np.isclose(max_val, min_val):
                normalized.loc[func_num] = 1.0
            else:
                normalized.loc[func_num] = (max_val - values) / (max_val - min_val)
            normalized.loc[func_num] = normalized.loc[func_num].fillna(0.0)

        theta = np.linspace(0, 2 * np.pi, len(funcs), endpoint=False)
        theta_closed = np.append(theta, theta[0])
        fig = plt.figure(figsize=(12, 12), facecolor="white")
        ax = plt.subplot(111, polar=True)
        ax.set_facecolor("white")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.grid(color="lightgray", linestyle="--", linewidth=0.6, alpha=0.8)

        for idx, algo in enumerate(algos):
            values = normalized[algo].to_numpy(dtype=float).tolist()
            values.append(values[0])
            ax.plot(theta_closed, values, color=cmap.get(algo), linewidth=3.0 if algo == "SCLPSO" else 2.0,
                    linestyle=line_styles[idx % len(line_styles)], label=algo)
            ax.fill(theta_closed, values, color=cmap.get(algo), alpha=0.05)

        ax.set_xticks(theta)
        ax.set_xticklabels([f"F{i}" for i in funcs], fontproperties=prop, fontsize=30, fontweight='bold')
        ax.set_yticks([0.2, 0.5, 0.8])
        ax.set_yticklabels(["0.2", "0.5", "0.8"], fontproperties=prop, fontsize=30, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title(f"CEC2017 {int(dim)}D Normalized Performance Radar", fontproperties=prop, fontsize=30, fontweight='bold', pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), prop=prop, fontsize=30)
        path = os.path.join(out_root, _safe_plot_name(final, f"SCLPSO_CEC2017_{int(dim)}D_CEC_Radar_Chart.jpg"))
        plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_wilcoxon_rank_sum_heatmap(raw_df: pd.DataFrame, out_root: str, final: bool = True) -> Optional[str]:
    paths = wilcoxon_rank_sum_vs_sclpso(raw_df, out_root, final=final)
    suffix = "" if final else "_partial"
    test_path = os.path.join(out_root, f"wilcoxon_rank_sum_vs_SCLPSO{suffix}.csv")
    if not os.path.exists(test_path):
        return None
    test_df = pd.read_csv(test_path)
    if test_df.empty:
        return None
    prop = configure_plot_style()
    fig_paths = []
    for dim, df_dim in test_df.groupby("dimension"):
        competitors = [a for a in CFG.ALGORITHMS if a != "SCLPSO" and a in set(df_dim["competitor"])]
        functions = sorted(df_dim["function"].unique())
        if not competitors or not functions:
            continue
        mat = np.zeros((len(competitors), len(functions)), dtype=float)
        for i, comp in enumerate(competitors):
            for j, func_num in enumerate(functions):
                row = df_dim[(df_dim["competitor"] == comp) & (df_dim["function"] == func_num)]
                if row.empty:
                    mat[i, j] = np.nan
                else:
                    sign = str(row.iloc[0].get("sign", "="))
                    mat[i, j] = 1.0 if sign == "+" else (-1.0 if sign == "-" else 0.0)
        fig, ax = plt.subplots(figsize=(max(12, 0.45 * len(functions)), max(5, 0.45 * len(competitors))))
        im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(np.arange(len(functions)))
        ax.set_xticklabels([f"F{int(f)}" for f in functions], rotation=90, fontproperties=prop, fontsize=30)
        ax.set_yticks(np.arange(len(competitors)))
        ax.set_yticklabels(competitors, fontproperties=prop, fontsize=30)
        ax.set_title(f"CEC2017 {int(dim)}D SCLPSO Significance Heatmap", fontproperties=prop, fontsize=30, fontweight='bold')
        ax.set_xlabel("Function", fontproperties=prop, fontsize=30, fontweight='bold')
        ax.set_ylabel("Competitor", fontproperties=prop, fontsize=30, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["SCLPSO worse", "n.s.", "SCLPSO better"])
        plt.tight_layout()
        path = os.path.join(out_root, _safe_plot_name(final, f"SCLPSO_Wilcoxon_RankSum_Heatmap_CEC2017_{int(dim)}D.jpg"))
        plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        fig_paths.append(path)
    return fig_paths[0] if fig_paths else None


def plot_friedman_average_rank_bar(summary_df: pd.DataFrame, out_root: str, final: bool = True) -> Optional[str]:
    friedman_df = compute_friedman_average_ranks(summary_df, out_root, final=final)
    if friedman_df.empty:
        return None
    prop = configure_plot_style()
    plot_df = (
        friedman_df.groupby("algorithm", as_index=False)["avg_rank"]
        .mean()
        .sort_values("avg_rank", ascending=True)
    )
    cmap = _algorithm_color_map(list(plot_df["algorithm"]))
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(plot_df))
    bars = ax.barh(y, plot_df["avg_rank"], color=[cmap[a] for a in plot_df["algorithm"]], alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["algorithm"], fontproperties=prop, fontsize=30)
    ax.invert_yaxis()
    ax.set_xlabel("Friedman Average Rank", fontproperties=prop, fontsize=30, fontweight='bold')
    ax.set_title("Friedman Average Ranks on CEC2017", fontproperties=prop, fontsize=30, fontweight='bold')
    ax.grid(True, alpha=0.3, axis="x")
    for bar, rank_value in zip(bars, plot_df["avg_rank"]):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{rank_value:.3f}", ha="left", va="center", fontproperties=prop, fontsize=30, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_Friedman_Average_Rank_Bar_CEC2017.jpg"))
    plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
    plt.close()
    return path


def generate_visual_outputs(raw_df: pd.DataFrame,
                            summary_df: pd.DataFrame,
                            rank_df: pd.DataFrame,
                            wtl_df: pd.DataFrame,
                            out_root: str,
                            final: bool = True) -> List[str]:
    paths: List[str] = []
    for table_func, args in [
        (save_mean_std_reports, (raw_df, out_root, final)),
        (wilcoxon_rank_sum_vs_sclpso, (raw_df, out_root, final)),
    ]:
        try:
            generated = table_func(*args)
            if generated:
                paths.extend(generated)
        except Exception as e:
            print(f"[WARNING] Failed to generate table with {table_func.__name__}: {e}", flush=True)

    for plotter, args in [
        (plot_overall_average_rankings, (rank_df, out_root, final)),
        (plot_friedman_average_rank_bar, (summary_df, out_root, final)),
        (plot_dimension_average_rankings, (rank_df, out_root, final)),
        (plot_category_rankings, (summary_df, out_root, final)),
        (plot_win_tie_loss, (wtl_df, out_root, final)),
        (plot_convergence_curves, (raw_df, out_root, final)),
        (generate_algorithm_legend, (out_root, final)),
        (plot_average_rank_radar, (summary_df, out_root, final)),
        (plot_wilcoxon_rank_sum_heatmap, (raw_df, out_root, final)),
        (plot_convergence_curves_by_function, (raw_df, out_root, final)),
        (plot_boxplots_by_function, (raw_df, out_root, final)),
    ]:
        try:
            generated = plotter(*args)
            if isinstance(generated, list):
                paths.extend(generated)
            elif generated:
                paths.append(generated)
        except Exception as e:
            print(f"[WARNING] Failed to generate output with {plotter.__name__}: {e}", flush=True)
    return paths


def analyze_results(out_root: str, raw_path: str):
    configure_plot_style()
    df = pd.read_csv(raw_path)
    df = _filter_plot_algorithms(df)
    df_ok = df[(df["status"] == "ok") & np.isfinite(df["error"])].copy()
    if df_ok.empty:
        print("No successful results to analyze.")
        return

    summary_df = summarize_raw(df_ok)
    summary_path = os.path.join(out_root, "summary_error_by_dim_func_algorithm.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    rank_df = ranking_by_dimension(summary_df)
    rank_path = os.path.join(out_root, "ranking_by_function.csv")
    rank_df.to_csv(rank_path, index=False, encoding="utf-8-sig")

    avg_rank = (
        rank_df.groupby(["dimension", "algorithm"], as_index=False)["rank"]
        .mean()
        .sort_values(["dimension", "rank"])
    )
    avg_rank.to_csv(os.path.join(out_root, "average_ranking_by_dimension.csv"), index=False, encoding="utf-8-sig")

    wtl_df = win_tie_loss_against_sclpso(summary_df, target="SCLPSO")
    wtl_df.to_csv(os.path.join(out_root, "win_tie_loss_vs_SCLPSO.csv"), index=False, encoding="utf-8-sig")


    compute_friedman_average_ranks(summary_df, out_root, final=True)
    wilcoxon_rank_sum_vs_sclpso(df_ok, out_root, final=True)

    plot_paths = generate_visual_outputs(df_ok, summary_df, rank_df, wtl_df, out_root, final=True)

    print("Analysis files saved to:", out_root)
    print("Top average ranks:")
    print(avg_rank.groupby("dimension").head(len(CFG.ALGORITHMS)).to_string(index=False))
    if plot_paths:
        print(f"Generated visualization/table outputs: {len(plot_paths)} files")
        for p in plot_paths[:25]:
            print(f"  {p}")
        if len(plot_paths) > 25:
            print(f"  ... {len(plot_paths) - 25} additional figures/tables saved under output subdirectories")


# Main entry
def main():
    os.makedirs(CFG.OUT_ROOT, exist_ok=True)
    raw_path = os.path.join(CFG.OUT_ROOT, "raw_results.csv")
    pickle_path = os.path.join(CFG.OUT_ROOT, "raw_results.pkl")
    config_path = os.path.join(CFG.OUT_ROOT, "experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, ensure_ascii=False, indent=2)


    _load_cec2017_wrapper_class()

    dims, funcs, algos, runs = _current_settings()
    print("=" * 100)
    print("SCLPSO vs traditional algorithms on CEC2017")
    print("Output:", CFG.OUT_ROOT)
    print("Dimensions:", dims)
    print("Functions:", funcs)
    print("Algorithms:", algos)
    print("Runs:", runs, "Pop:", CFG.POP_SIZE, "Budget mode:", CFG.BUDGET_MODE)
    print("maxFEs:", {d: max_fes_for_dim(d) for d in dims})
    print("Parallel:", CFG.USE_PARALLEL, "Processes:", CFG.MAX_PROCESSES)
    print("=" * 100)

    all_tasks = make_tasks()
    existing_df = load_existing_results(raw_path)
    rows: List[Dict[str, Any]] = []
    done_keys = set()
    if not existing_df.empty:
        existing_rows = existing_df.to_dict("records")
        rows.extend(existing_rows)
        done_keys = {task_key(r) for r in existing_rows if str(r.get("status", "")) == "ok"}
        print(f"Resume enabled: loaded {len(existing_rows)} rows, {len(done_keys)} completed OK tasks.")

    tasks = [t for t in all_tasks if t not in done_keys]
    total = len(all_tasks)
    print(f"Tasks remaining: {len(tasks)} / {total}")

    start_total = time.time()
    finished_since_save = 0

    if CFG.USE_PARALLEL and len(tasks) > 1:
        n_proc = max(1, min(CFG.MAX_PROCESSES, len(tasks)))
        with Pool(processes=n_proc, maxtasksperchild=CFG.TASKS_PER_CHILD) as pool:
            for idx, row in enumerate(pool.imap_unordered(run_task, tasks, chunksize=CFG.CHUNKSIZE), start=1):
                rows.append(row)
                finished_since_save += 1
                status = row.get("status")
                print(
                    f"[{idx}/{len(tasks)}] D{row['dimension']} F{row['function']} {row['algorithm']} "
                    f"run={row['run']} status={status} error={row.get('error', np.inf):.3e} "
                    f"time={row.get('elapsed_sec', 0.0):.1f}s",
                    flush=True,
                )
                if finished_since_save >= CFG.SAVE_EVERY:
                    save_results(rows, raw_path, pickle_path)
                    finished_since_save = 0
                    elapsed = (time.time() - start_total) / 60.0
                    print(f"Partial results saved. Elapsed: {elapsed:.1f} min")
    else:
        for idx, task in enumerate(tasks, start=1):
            row = run_task(task)
            rows.append(row)
            finished_since_save += 1
            print(
                f"[{idx}/{len(tasks)}] D{row['dimension']} F{row['function']} {row['algorithm']} "
                f"run={row['run']} status={row.get('status')} error={row.get('error', np.inf):.3e} "
                f"time={row.get('elapsed_sec', 0.0):.1f}s",
                flush=True,
            )
            if finished_since_save >= CFG.SAVE_EVERY:
                save_results(rows, raw_path, pickle_path)
                finished_since_save = 0

    save_results(rows, raw_path, pickle_path)
    print(f"All raw results saved: {raw_path}")
    print(f"Total elapsed: {(time.time() - start_total) / 60.0:.1f} min")

    analyze_results(CFG.OUT_ROOT, raw_path)
    print("Done.")


if __name__ == "__main__":
    main()
