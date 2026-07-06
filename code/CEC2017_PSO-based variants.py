# -*- coding: utf-8 -*-


import os



os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
import math
import time
import pickle
import traceback
from dataclasses import dataclass, asdict
from ctypes import CDLL, POINTER, c_int, c_double
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional, Any

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


# Configuration
@dataclass
class ExperimentConfig:



    SO_PATH: str = "cec17_func.so"


    OUT_ROOT: str = "/public/home/sszzkli/.local/SCLPSO_REVISED/CEC2017_SCLPSO_NOSTATE_NOESCAPE_EAPSO_DBCD_OLPSO_WITH_F2_TIMES30_FONT"



    FONT_PATH: str = "/public/home/sszzkli/.local/Times New Roman.ttf"
    PLOT_DPI: int = 300


    DIMENSIONS: Tuple[int, ...] = (10, 30, 50, 100)



    EXCLUDE_F2: bool = False


    N_RUNS: int = 51


    MAX_FES_FACTOR: int = 10000


    LOWER_BOUND: float = -100.0
    UPPER_BOUND: float = 100.0



    POP_SIZE: int = 30



    ALGORITHMS: Tuple[str, ...] = (
        "SCLPSO",
        "PSO",
        "APSO",
        "FIPS",
        "DMS_PSO",
        "QPSO",
        "FDR_PSO",
        "EAPSO",
        "PSO_DBCD",
        "OLPSO",
    )




    USE_PARALLEL: bool = True
    MAX_PROCESSES: int = min(60, cpu_count() or 1)
    TASKS_PER_CHILD: int = 25
    CHUNKSIZE: int = 1
    SHUFFLE_TASKS: bool = True
    VERBOSE_TASKS: bool = False



    CURVE_POINTS: int = 200


    TEST_MODE: bool = False


    BASE_SEED: int = 20260521


    SAVE_EVERY: int = 50


CFG = ExperimentConfig()


# CEC2017 interface
class CEC17DLLWrapper:


    def __init__(self, dim: int, func_num: int, so_path: str):
        self.dim = int(dim)
        self.func_num = int(func_num)

        if not os.path.exists(so_path):
            raise FileNotFoundError(
                f"CEC2017 shared library not found: {so_path}\n"
                f"Please put cec17_func.so / cec17_func.dll in this path or set CFG.SO_PATH."
            )

        self.so_path = os.path.abspath(so_path)
        self.so = CDLL(self.so_path)

        self.so.cec17_test_func.argtypes = [
            POINTER(c_double),
            POINTER(c_double),
            c_int,
            c_int,
            c_int
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


class BudgetedCEC17:


    def __init__(self, dim: int, func_num: int, max_fes: int, so_path: str):
        self.wrapper = CEC17DLLWrapper(dim=dim, func_num=func_num, so_path=so_path)
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

        n = pop.shape[0]
        if n <= 0:
            return np.empty(0, dtype=np.float64)

        if self.fes >= self.max_fes:
            return np.full(n, np.inf, dtype=np.float64)

        allowed = min(n, self.max_fes - self.fes)
        out = np.full(n, np.inf, dtype=np.float64)
        if allowed > 0:
            out[:allowed] = self.wrapper.evaluate_population(pop[:allowed])
            self.fes += allowed
        return out

    def eval_one(self, x: np.ndarray) -> float:
        return float(self.eval_pop(np.asarray(x, dtype=np.float64).reshape(1, -1))[0])


# Utilities
def make_functions(exclude_f2: bool = False) -> List[int]:


    funcs = list(range(1, 31))
    if exclude_f2:
        funcs = [f for f in funcs if f != 2]
    return funcs


def format_function_list(funcs: List[int]) -> str:
    if funcs == [1] + list(range(3, 31)):
        return "F1 and F3-F30 (29 functions; F2 excluded for backward compatibility)"
    if funcs == list(range(1, 31)):
        return "F1-F30 (30 functions; F2 included)"
    return str(funcs)


def bounds_tuple() -> Tuple[float, float]:
    return float(CFG.LOWER_BOUND), float(CFG.UPPER_BOUND)


def stable_algorithm_index(name: str) -> int:

    names = list(CFG.ALGORITHMS)
    if name in names:
        return names.index(name) + 1
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


def halton_chaos_init(pop_size: int, dim: int, bounds: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:

    lb, ub = bounds
    sampler_seed = int(rng.integers(1, 2**31 - 1))
    sampler = qmc.Halton(d=dim, scramble=True, seed=sampler_seed)
    halton = qmc.scale(sampler.random(n=pop_size), lb, ub)

    chaos = rng.random((pop_size, dim))

    for _ in range(5):
        chaos = 4.0 * chaos * (1.0 - chaos)
    chaos_points = lb + chaos * (ub - lb)

    pop = 0.7 * halton + 0.3 * chaos_points
    return np.clip(pop, lb, ub).astype(np.float64)


def population_diversity(pop: np.ndarray, bounds: Tuple[float, float]) -> float:
    lb, ub = bounds
    return float(np.mean(np.std(pop, axis=0)) / (ub - lb + 1e-12))


def levy_step(dim: int, beta_min: float, beta_max: float, progress: float, rng: np.random.Generator) -> np.ndarray:
    beta = beta_min + (beta_max - beta_min) * progress
    numerator = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    denominator = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma = (numerator / denominator) ** (1.0 / beta)

    u = rng.normal(0.0, sigma, dim)
    v = rng.normal(0.0, 1.0, dim)
    eta = 0.3 * (1.0 - progress) + 0.05
    return eta * u / (np.abs(v) ** (1.0 / beta) + 1e-12)


class CurveRecorder:


    def __init__(self, max_fes: int, n_points: int = 200):
        self.max_fes = int(max_fes)
        self.n_points = int(max(10, n_points))
        self.interval = max(1, self.max_fes // self.n_points)
        self.next_fes = 0
        self.fes: List[int] = []
        self.best: List[float] = []
        self.diversity: List[float] = []
        self.state: List[str] = []

    def update(self, fes: int, best_value: float, diversity: Optional[float] = None, state: str = ""):
        if fes >= self.next_fes or fes >= self.max_fes:
            self.fes.append(int(fes))
            self.best.append(float(best_value))
            if diversity is not None:
                self.diversity.append(float(diversity))
            else:
                self.diversity.append(float("nan"))
            self.state.append(str(state))
            self.next_fes += self.interval

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fes": self.fes,
            "best": self.best,
            "diversity": self.diversity,
            "state": self.state,
        }


# SCLPSO parameters
SCLPSO_PARAMS: Dict[str, float] = {
















    "SCLPSO_NINIT_FACTOR": 6.0,
    "SCLPSO_MAX_POP": 720,
    "SCLPSO_NMIN": 30,


    "CL_REFRESH_GAP": 7,
    "CL_C": 1.49445,
    "GBEST_C_MIN": 0.05,
    "GBEST_C_MAX": 0.85,


    "SHADE_H": 6,
    "SHADE_P": 0.11,
    "SHADE_ARCHIVE_RATE": 2.6,
    "SHADE_RATE_EXPLORATION": 1.00,
    "SHADE_RATE_EXPLOITATION": 0.80,
    "SHADE_RATE_STAGNATION": 0.55,
    "SHADE_RATE_ESCAPE": 0.35,


    "DVA_MU": 18,
    "DVA_CANDIDATES_EXPLORATION": 0,
    "DVA_CANDIDATES_EXPLOITATION": 3,
    "DVA_CANDIDATES_STAGNATION": 2,
    "DVA_CANDIDATES_ESCAPE": 1,


    "RESTART_THRESHOLD": 14,
    "LEVY_BETA_MIN": 1.1980903890921581,
    "LEVY_BETA_MAX": 1.7346654255368175,
    "OBL_RATIO": 0.20,
    "WORST_RESET_RATIO": 0.15,


    "DIVERSITY_LOW": 1e-5,
    "DIVERSITY_MID": 0.08,
    "IMPROVEMENT_EPS": 1e-12,
    "IMPROVEMENT_WINDOW": 5,



    "DISABLE_STATE_GUIDANCE": True,
    "DISABLE_DVA": True,
    "DISABLE_ESCAPE": True,
}


class DiagonalVarianceAdapter:


    def __init__(self, dim: int, mu: int):
        self.dim = int(dim)
        self.mu = int(mu)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.path_sigma = np.zeros(dim, dtype=np.float64)
        self.sigma = 1.0


        self.c_var = 2.0 / ((dim + 2.0) ** 1.5)
        self.c_sigma = 0.3

    def update(self, population: np.ndarray, fitness: np.ndarray):
        pop_size = population.shape[0]
        mu = max(1, min(self.mu, pop_size))
        idx = np.argsort(fitness)[:mu]
        elite = population[idx]

        weights = np.array([np.log(mu + 0.5) - np.log(i + 1.0) for i in range(mu)], dtype=np.float64)
        weights = weights / np.sum(weights)

        old_mean = self.mean.copy()
        new_mean = np.sum(weights[:, None] * elite, axis=0)

        y = (new_mean - old_mean) / max(self.sigma, 1e-20)
        self.path_sigma = (
            (1.0 - self.c_sigma) * self.path_sigma
            + np.sqrt(self.c_sigma * (2.0 - self.c_sigma)) * y
        )

        z = (elite - old_mean[None, :]) / max(self.sigma, 1e-20)
        weighted_z2 = np.sum(weights[:, None] * (z ** 2), axis=0)

        self.var = (1.0 - self.c_var) * self.var + self.c_var * weighted_z2
        self.var = np.clip(self.var, 1e-12, 1e12)

        path_norm = np.linalg.norm(self.path_sigma)
        if path_norm > 1e3:
            self.path_sigma *= 1e3 / path_norm
            path_norm = 1e3

        self.sigma *= np.exp((path_norm / np.sqrt(self.dim) - 1.0) * self.c_sigma / 2.0)
        self.sigma = float(np.clip(self.sigma, 1e-20, 1e20))
        self.mean = new_mean

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self.mean + self.sigma * np.sqrt(self.var) * rng.normal(size=self.dim)


def diagnose_search_state(
    diversity: float,
    stagnation: int,
    recent_improvement: float,
    params: Dict[str, float],
) -> str:


    if stagnation >= 2 * int(params["RESTART_THRESHOLD"]):
        return "escape"
    if stagnation >= int(params["RESTART_THRESHOLD"]) or diversity < params["DIVERSITY_LOW"]:
        return "stagnation"
    if recent_improvement <= params["IMPROVEMENT_EPS"] and diversity >= params["DIVERSITY_MID"]:
        return "exploration"
    return "exploitation"


def sclpso_revised(
    dim: int,
    func_num: int,
    pop_size: int,
    max_fes: int,
    bounds: Tuple[float, float],
    so_path: str,
    seed: int,
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:


    if params is None:
        params = SCLPSO_PARAMS
    params = dict(params)


    params["DISABLE_STATE_GUIDANCE"] = True
    params["DISABLE_DVA"] = True
    params["DISABLE_ESCAPE"] = True

    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim=dim, func_num=func_num, max_fes=max_fes, so_path=so_path)
    recorder = CurveRecorder(max_fes=max_fes, n_points=CFG.CURVE_POINTS)

    lb, ub = bounds







    n_min = max(4, int(params["SCLPSO_NMIN"]))
    n_min = max(n_min, min(pop_size, int(params["SCLPSO_MAX_POP"])))
    n_init = int(round(float(params["SCLPSO_NINIT_FACTOR"]) * dim))
    n_init = max(pop_size, n_init, n_min)
    n_init = min(int(params["SCLPSO_MAX_POP"]), n_init)

    n_init = min(n_init, max(n_min, max_fes // 10))
    pop_size_eff = n_init

    particles = halton_chaos_init(pop_size_eff, dim, bounds, rng)
    velocities = np.zeros((pop_size_eff, dim), dtype=np.float64)

    fitness = evaluator.eval_pop(particles)
    pbest_pos = particles.copy()
    pbest_val = fitness.copy()

    gbest_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[gbest_idx].copy()
    best_fit = float(pbest_val[gbest_idx])


    pc = _clpso_learning_probability(pop_size_eff)
    refresh_gap = int(params["CL_REFRESH_GAP"])
    no_improve = np.zeros(pop_size_eff, dtype=np.int64)
    exemplars = np.tile(np.arange(pop_size_eff)[:, None], (1, dim))

    def refresh_particle(i: int):

        n = particles.shape[0]
        if n <= 1:
            exemplars[i, :] = i
            return
        candidates_except_i = [j for j in range(n) if j != i]
        all_own = True
        for d in range(dim):
            if rng.random() < pc[i] and len(candidates_except_i) >= 2:
                a, b = rng.choice(candidates_except_i, size=2, replace=False)
                winner = int(a if pbest_val[a] < pbest_val[b] else b)
                exemplars[i, d] = winner
                all_own = False
            else:
                exemplars[i, d] = i
        if all_own and len(candidates_except_i) >= 2:
            d = int(rng.integers(0, dim))
            a, b = rng.choice(candidates_except_i, size=2, replace=False)
            exemplars[i, d] = int(a if pbest_val[a] < pbest_val[b] else b)

    def refresh_all_exemplars():
        for ii in range(particles.shape[0]):
            refresh_particle(ii)

    refresh_all_exemplars()


    h_size = int(params["SHADE_H"])
    m_cr = np.full(h_size, 0.5, dtype=np.float64)
    m_f = np.full(h_size, 0.5, dtype=np.float64)
    memory_pos = 0
    archive = np.empty((0, dim), dtype=np.float64)

    dva = DiagonalVarianceAdapter(dim=dim, mu=int(params["DVA_MU"]))

    stagnation = 0
    iteration = 0
    best_history = [best_fit]

    while evaluator.fes < max_fes:
        n = particles.shape[0]
        progress = evaluator.fes / max_fes
        diversity = population_diversity(particles, bounds)


        win = int(params["IMPROVEMENT_WINDOW"])
        if len(best_history) > win:
            previous = best_history[-win - 1]
            recent_improvement = abs(previous - best_fit) / (abs(previous) + 1e-12)
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
        c_g_min = float(params["GBEST_C_MIN"])
        c_g_max = float(params["GBEST_C_MAX"])
        if state == "exploration":
            c_g = c_g_min * progress
        elif state == "exploitation":
            c_g = c_g_min + (c_g_max - c_g_min) * progress
        else:
            c_g = c_g_max

        velocities = (
            w * velocities
            + c_cl * rng.random((n, dim)) * (exemplar_pos - particles)
            + c_g * rng.random((n, dim)) * (gbest_pos[None, :] - particles)
        )
        velocities = clamp_velocity(velocities, bounds, 0.2)
        particles = reflect_bounds(particles + velocities, bounds)

        if evaluator.remaining() > 0:
            new_fit = evaluator.eval_pop(particles)
            improved = new_fit < pbest_val
            pbest_pos[improved] = particles[improved]
            pbest_val[improved] = new_fit[improved]
            no_improve[improved] = 0
            no_improve[~improved] += 1

            current_idx = int(np.argmin(pbest_val))
            current_best = float(pbest_val[current_idx])
            if current_best < best_fit:
                best_fit = current_best
                gbest_pos = pbest_pos[current_idx].copy()
                stagnation = 0
            else:
                stagnation += 1





        if evaluator.remaining() > 0 and n >= 4:
            if state == "exploration":
                shade_rate = float(params["SHADE_RATE_EXPLORATION"])
            elif state == "exploitation":
                shade_rate = float(params["SHADE_RATE_EXPLOITATION"])
            elif state == "stagnation":
                shade_rate = float(params["SHADE_RATE_STAGNATION"])
            else:
                shade_rate = float(params["SHADE_RATE_ESCAPE"])

            n_trials = max(1, int(round(shade_rate * n)))
            n_trials = min(n_trials, evaluator.remaining(), n)
            trial_indices = rng.choice(n, size=n_trials, replace=False)

            order = np.argsort(pbest_val)
            p_num = max(2, int(round(float(params["SHADE_P"]) * n)))
            p_num = min(p_num, n)
            top_indices = order[:p_num]

            union = particles if archive.shape[0] == 0 else np.vstack([particles, archive])
            union_size = union.shape[0]

            trials = np.empty((n_trials, dim), dtype=np.float64)
            cr_values = np.empty(n_trials, dtype=np.float64)
            f_values = np.empty(n_trials, dtype=np.float64)

            for kk, i in enumerate(trial_indices):
                mem_idx = int(rng.integers(0, h_size))

                if np.isnan(m_cr[mem_idx]):
                    cr = 0.0
                else:
                    cr = float(np.clip(rng.normal(m_cr[mem_idx], 0.1), 0.0, 1.0))
                f = _sample_lshade_F(m_f[mem_idx], rng)

                cr_values[kk] = cr
                f_values[kk] = f

                pbest_idx = int(rng.choice(top_indices))

                candidates_r1 = np.arange(n)
                candidates_r1 = candidates_r1[candidates_r1 != i]
                r1_idx = int(rng.choice(candidates_r1))


                while True:
                    r2_idx = int(rng.integers(0, union_size))
                    if r2_idx == i:
                        continue
                    if r2_idx == r1_idx:
                        continue
                    break



                mutant = (
                    particles[i]
                    + f * (pbest_pos[pbest_idx] - particles[i])
                    + f * (particles[r1_idx] - union[r2_idx])
                )
                mutant = _shade_boundary_correction(mutant, particles[i], bounds)

                cross = rng.random(dim) <= cr
                cross[int(rng.integers(0, dim))] = True
                trials[kk] = np.where(cross, mutant, particles[i])

            trial_fit = evaluator.eval_pop(trials)

            s_cr: List[float] = []
            s_f: List[float] = []
            delta_f: List[float] = []
            new_archive_members: List[np.ndarray] = []

            for kk, i in enumerate(trial_indices):
                if not np.isfinite(trial_fit[kk]):
                    continue


                if trial_fit[kk] <= fitness[i]:
                    if trial_fit[kk] < fitness[i]:
                        new_archive_members.append(particles[i].copy())
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

            if new_archive_members:
                archive = np.vstack([archive, np.asarray(new_archive_members, dtype=np.float64)])

            archive_max = max(0, int(round(float(params["SHADE_ARCHIVE_RATE"]) * n)))
            if archive.shape[0] > archive_max:
                keep = rng.choice(archive.shape[0], size=archive_max, replace=False)
                archive = archive[keep]

            if len(s_f) > 0 and len(s_cr) > 0:
                s_cr_arr = np.asarray(s_cr, dtype=np.float64)
                s_f_arr = np.asarray(s_f, dtype=np.float64)
                df_arr = np.asarray(delta_f, dtype=np.float64)
                if float(np.sum(df_arr)) <= 0.0:
                    weights = np.full_like(df_arr, 1.0 / len(df_arr))
                else:
                    weights = df_arr / np.sum(df_arr)

                if np.isnan(m_cr[memory_pos]) or np.max(s_cr_arr) == 0.0:
                    m_cr[memory_pos] = np.nan
                else:
                    m_cr[memory_pos] = _weighted_lehmer_mean(s_cr_arr, weights)
                m_f[memory_pos] = _weighted_lehmer_mean(s_f_arr, weights)
                memory_pos = (memory_pos + 1) % h_size





        if evaluator.remaining() > 0 and not bool(params.get("DISABLE_DVA", True)):
            if state == "exploration":
                n_cand = int(params["DVA_CANDIDATES_EXPLORATION"])
            elif state == "exploitation":
                n_cand = int(params["DVA_CANDIDATES_EXPLOITATION"])
            elif state == "stagnation":
                n_cand = int(params["DVA_CANDIDATES_STAGNATION"])
            else:
                n_cand = int(params["DVA_CANDIDATES_ESCAPE"])

            n_cand = min(n_cand, evaluator.remaining())
            if n_cand > 0:
                dva.update(pbest_pos, pbest_val)
                cand = np.vstack([reflect_bounds(dva.sample(rng), bounds) for _ in range(n_cand)])
                cand_fit = evaluator.eval_pop(cand)

                for kk in range(n_cand):
                    worst_idx = int(np.argmax(pbest_val))
                    if cand_fit[kk] < pbest_val[worst_idx]:
                        particles[worst_idx] = cand[kk]
                        fitness[worst_idx] = cand_fit[kk]
                        pbest_pos[worst_idx] = cand[kk]
                        pbest_val[worst_idx] = cand_fit[kk]
                        no_improve[worst_idx] = 0
                        refresh_particle(worst_idx)
                        if cand_fit[kk] < best_fit:
                            best_fit = float(cand_fit[kk])
                            gbest_pos = cand[kk].copy()
                            stagnation = 0





        if evaluator.remaining() > 0 and not bool(params.get("DISABLE_ESCAPE", True)) and (state in ("stagnation", "escape")):
            reset_num = max(1, int(round(float(params["WORST_RESET_RATIO"]) * n)))
            reset_num = min(reset_num, evaluator.remaining(), n)
            worst_indices = np.argsort(pbest_val)[-reset_num:]

            candidates = []
            for idx in worst_indices:
                if state == "escape" and rng.random() < 0.5:
                    step = levy_step(
                        dim,
                        beta_min=float(params["LEVY_BETA_MIN"]),
                        beta_max=float(params["LEVY_BETA_MAX"]),
                        progress=progress,
                        rng=rng,
                    )
                    x_new = gbest_pos + step * (1.0 - progress)
                else:

                    x_new = lb + ub - particles[idx]

                    x_new = 0.85 * x_new + 0.15 * (gbest_pos + rng.normal(0, 0.1 * (ub - lb), size=dim))
                candidates.append(reflect_bounds(x_new, bounds))

            candidates = np.asarray(candidates, dtype=np.float64)
            cand_fit = evaluator.eval_pop(candidates)

            for kk, idx in enumerate(worst_indices):
                particles[idx] = candidates[kk]
                fitness[idx] = cand_fit[kk]

                if cand_fit[kk] < pbest_val[idx]:
                    pbest_pos[idx] = candidates[kk]
                    pbest_val[idx] = cand_fit[kk]
                    if cand_fit[kk] < best_fit:
                        best_fit = float(cand_fit[kk])
                        gbest_pos = candidates[kk].copy()
                no_improve[idx] = 0
                refresh_particle(int(idx))
            stagnation = 0




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
            refresh_all_exemplars()

            archive_max = max(0, int(round(float(params["SHADE_ARCHIVE_RATE"]) * target_n)))
            if archive.shape[0] > archive_max:
                keep_a = rng.choice(archive.shape[0], size=archive_max, replace=False)
                archive = archive[keep_a]

        best_history.append(best_fit)
        recorder.update(evaluator.fes, best_fit, diversity=diversity, state=f"SCLPSO-noEscape-{state}")
        iteration += 1

    return {
        "best": float(best_fit),
        "fes": evaluator.fes,
        "curve": recorder.as_dict(),
    }


# Baseline algorithms
def pso_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    x = uniform_init(pop_size, dim, bounds, rng)
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
        c1 = 2.0
        c2 = 2.0
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest[None, :] - x)
        x = reflect_bounds(x + v, bounds)
        fit = evaluator.eval_pop(x)
        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        g_idx = int(np.argmin(pbest_fit))
        if pbest_fit[g_idx] < best:
            best = float(pbest_fit[g_idx])
            gbest = pbest[g_idx].copy()
        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="PSO")

    return {"best": best, "fes": evaluator.fes, "curve": recorder.as_dict()}


def _mean_distance_matrix(pop: np.ndarray) -> np.ndarray:

    n = pop.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    diff = pop[:, None, :] - pop[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return np.sum(dist, axis=1) / max(1, n - 1)


def _apso_memberships(f: float) -> Dict[str, float]:

    f = float(np.clip(f, 0.0, 1.0))

    if f <= 0.4:
        s1 = 0.0
    elif f <= 0.6:
        s1 = 5.0 * f - 2.0
    elif f <= 0.7:
        s1 = 1.0
    elif f <= 0.8:
        s1 = -10.0 * f + 8.0
    else:
        s1 = 0.0

    if f <= 0.2:
        s2 = 0.0
    elif f <= 0.3:
        s2 = 10.0 * f - 2.0
    elif f <= 0.4:
        s2 = 1.0
    elif f <= 0.6:
        s2 = -5.0 * f + 3.0
    else:
        s2 = 0.0

    if f <= 0.1:
        s3 = 1.0
    elif f <= 0.3:
        s3 = -5.0 * f + 1.5
    else:
        s3 = 0.0

    if f <= 0.7:
        s4 = 0.0
    elif f <= 0.9:
        s4 = 5.0 * f - 3.5
    else:
        s4 = 1.0
    return {"S1": max(0.0, s1), "S2": max(0.0, s2), "S3": max(0.0, s3), "S4": max(0.0, s4)}


def _apso_classify_state(f: float, previous_state: str = "S1") -> str:


    mus = _apso_memberships(f)
    max_mu = max(mus.values())
    candidates = [k for k, v in mus.items() if abs(v - max_mu) <= 1e-12]
    if len(candidates) == 1:
        return candidates[0]
    order = ["S1", "S2", "S3", "S4"]
    if previous_state in candidates:
        return previous_state
    next_state = order[(order.index(previous_state) + 1) % len(order)] if previous_state in order else "S1"
    if next_state in candidates:
        return next_state
    return candidates[0]


def _apso_update_c1_c2(c1: float, c2: float, state: str, rng: np.random.Generator) -> Tuple[float, float]:

    delta = float(rng.uniform(0.05, 0.10))
    if state == "S1":
        c1 += delta
        c2 -= delta
    elif state == "S2":
        c1 += 0.5 * delta
        c2 -= 0.5 * delta
    elif state == "S3":
        c1 += 0.5 * delta
        c2 += 0.5 * delta
    elif state == "S4":
        c1 -= delta
        c2 += delta

    c1 = float(np.clip(c1, 1.5, 2.5))
    c2 = float(np.clip(c2, 1.5, 2.5))
    s = c1 + c2
    if s > 4.0:
        c1, c2 = c1 / s * 4.0, c2 / s * 4.0
    elif s < 3.0:
        c1, c2 = c1 / s * 3.0, c2 / s * 3.0
    return float(c1), float(c2)


def apso_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    x = uniform_init(pop_size, dim, bounds, rng)
    v = np.zeros_like(x)
    fit = evaluator.eval_pop(x)
    pbest = x.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(pbest_fit))
    gbest = pbest[g_idx].copy()
    best = float(pbest_fit[g_idx])

    c1, c2 = 2.0, 2.0
    state = "S1"

    while evaluator.fes < max_fes:

        d_mean = _mean_distance_matrix(x)
        d_min, d_max = float(np.min(d_mean)), float(np.max(d_mean))
        g_idx = int(np.argmin(pbest_fit))
        dg = float(d_mean[g_idx])
        f = 0.0 if abs(d_max - d_min) < 1e-30 else (dg - d_min) / (d_max - d_min)
        f = float(np.clip(f, 0.0, 1.0))

        state = _apso_classify_state(f, state)
        w = 1.0 / (1.0 + 1.5 * math.exp(-2.6 * f))
        c1, c2 = _apso_update_c1_c2(c1, c2, state, rng)

        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest[None, :] - x)
        v = clamp_velocity(v, bounds, 0.2)
        x = reflect_bounds(x + v, bounds)
        fit = evaluator.eval_pop(x)

        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        g_idx = int(np.argmin(pbest_fit))
        if pbest_fit[g_idx] < best:
            best = float(pbest_fit[g_idx])
            gbest = pbest[g_idx].copy()


        if state == "S3" and evaluator.can_eval(1):
            progress = evaluator.fes / max_fes
            sigma = 1.0 - 0.9 * progress
            candidate = gbest.copy()
            d = int(rng.integers(0, dim))
            candidate[d] += (bounds[1] - bounds[0]) * rng.normal(0.0, sigma)
            candidate = reflect_bounds(candidate, bounds)
            cand_fit = evaluator.eval_one(candidate)
            if cand_fit < best:
                best = float(cand_fit)
                gbest = candidate.copy()
                worst_idx = int(np.argmax(pbest_fit))
                x[worst_idx] = candidate.copy()
                pbest[worst_idx] = candidate.copy()
                pbest_fit[worst_idx] = cand_fit
            else:

                worst_idx = int(np.argmax(fit))
                x[worst_idx] = candidate.copy()
                fit[worst_idx] = cand_fit
                pbest[worst_idx] = candidate.copy()
                pbest_fit[worst_idx] = cand_fit

        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state=f"APSO-{state}")

    return {"best": best, "fes": evaluator.fes, "curve": recorder.as_dict()}


def _clpso_learning_probability(pop_size: int) -> np.ndarray:

    if pop_size <= 1:
        return np.array([0.5], dtype=np.float64)
    i = np.arange(pop_size, dtype=np.float64)
    pc = 0.05 + 0.45 * (np.exp(10.0 * i / (pop_size - 1.0)) - 1.0) / (np.exp(10.0) - 1.0)
    return pc.astype(np.float64)


def clpso_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    x = uniform_init(pop_size, dim, bounds, rng)
    v = np.zeros_like(x)
    fit = evaluator.eval_pop(x)
    pbest = x.copy()
    pbest_fit = fit.copy()
    best = float(np.min(pbest_fit))

    pc = _clpso_learning_probability(pop_size)
    refresh_gap = 7
    no_improve = np.zeros(pop_size, dtype=np.int64)
    exemplars = np.tile(np.arange(pop_size)[:, None], (1, dim))

    def refresh_particle(i: int):
        all_own = True
        candidates_except_i = [j for j in range(pop_size) if j != i]
        for d in range(dim):
            if rng.random() < pc[i] and len(candidates_except_i) >= 2:
                a, b = rng.choice(candidates_except_i, size=2, replace=False)
                winner = a if pbest_fit[a] < pbest_fit[b] else b
                exemplars[i, d] = int(winner)
                all_own = False
            else:
                exemplars[i, d] = i

        if all_own and len(candidates_except_i) >= 2:
            d = int(rng.integers(0, dim))
            a, b = rng.choice(candidates_except_i, size=2, replace=False)
            exemplars[i, d] = int(a if pbest_fit[a] < pbest_fit[b] else b)

    for i in range(pop_size):
        refresh_particle(i)

    while evaluator.fes < max_fes:
        for i in range(pop_size):
            if no_improve[i] >= refresh_gap:
                refresh_particle(i)
                no_improve[i] = 0

        exemplar_pos = np.empty_like(x)
        dims = np.arange(dim)
        for i in range(pop_size):
            exemplar_pos[i] = pbest[exemplars[i], dims]

        progress = evaluator.fes / max_fes
        w = 0.9 - 0.5 * progress
        c = 1.49445
        v = w * v + c * rng.random((pop_size, dim)) * (exemplar_pos - x)
        v = clamp_velocity(v, bounds, 0.2)
        x = reflect_bounds(x + v, bounds)
        fit = evaluator.eval_pop(x)

        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        no_improve[improved] = 0
        no_improve[~improved] += 1

        current_best = float(np.min(pbest_fit))
        if current_best < best:
            best = current_best
        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="Classic_CLPSO")

    return {"best": best, "fes": evaluator.fes, "curve": recorder.as_dict()}


def _ring_neighbors(i: int, n: int, include_self: bool = False, radius: int = 1) -> List[int]:
    neigh = []
    for r in range(1, radius + 1):
        neigh.append((i - r) % n)
        neigh.append((i + r) % n)
    if include_self:
        neigh.append(i)

    return list(dict.fromkeys(neigh))


def fips_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    x = uniform_init(pop_size, dim, bounds, rng)
    v = np.zeros_like(x)
    fit = evaluator.eval_pop(x)
    pbest = x.copy()
    pbest_fit = fit.copy()
    best = float(np.min(pbest_fit))


    w = 0.729
    phi = 4.1
    include_self = False
    radius = 1
    neighborhoods = [_ring_neighbors(i, pop_size, include_self=include_self, radius=radius) for i in range(pop_size)]

    while evaluator.fes < max_fes:
        new_v = np.empty_like(v)
        for i in range(pop_size):
            neigh = neighborhoods[i]
            if not neigh:
                neigh = [i]
            influence = np.zeros(dim, dtype=np.float64)
            for m in neigh:
                gamma = rng.random(dim) * phi
                influence += gamma * (pbest[m] - x[i])
            influence /= float(len(neigh))
            new_v[i] = w * v[i] + influence
        v = clamp_velocity(new_v, bounds, 0.2)
        x = reflect_bounds(x + v, bounds)
        fit = evaluator.eval_pop(x)

        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        current_best = float(np.min(pbest_fit))
        if current_best < best:
            best = current_best
        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="FIPS")

    return {"best": best, "fes": evaluator.fes, "curve": recorder.as_dict()}

def de_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    F = 0.5
    CR = 0.9
    pop = uniform_init(pop_size, dim, bounds, rng)
    fit = evaluator.eval_pop(pop)
    best = float(np.min(fit))

    while evaluator.fes < max_fes:
        trial_list = []
        indices = []
        for i in range(pop_size):
            if len(trial_list) >= evaluator.remaining():
                break
            choices = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(choices, size=3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            cross = rng.random(dim) < CR
            cross[int(rng.integers(0, dim))] = True
            trial = np.where(cross, mutant, pop[i])
            trial = reflect_bounds(trial, bounds)
            trial_list.append(trial)
            indices.append(i)

        if not trial_list:
            break
        trials = np.asarray(trial_list, dtype=np.float64)
        trial_fit = evaluator.eval_pop(trials)
        for k, i in enumerate(indices):
            if trial_fit[k] < fit[i]:
                pop[i] = trials[k]
                fit[i] = trial_fit[k]
        current_best = float(np.min(fit))
        if current_best < best:
            best = current_best
        recorder.update(evaluator.fes, best, diversity=population_diversity(pop, bounds), state="DE")

    return {"best": best, "fes": evaluator.fes, "curve": recorder.as_dict()}


def bbpso_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    x = uniform_init(pop_size, dim, bounds, rng)
    fit = evaluator.eval_pop(x)
    pbest = x.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(pbest_fit))
    gbest = pbest[g_idx].copy()
    best = float(pbest_fit[g_idx])

    lb, ub = bounds
    min_std = 1e-12 * (ub - lb)

    while evaluator.fes < max_fes:
        n_samples = min(pop_size, evaluator.remaining())
        if n_samples <= 0:
            break



        order = rng.permutation(pop_size)[:n_samples]
        center = 0.5 * (pbest[order] + gbest[None, :])
        sigma = np.maximum(np.abs(pbest[order] - gbest[None, :]), min_std)
        x_new = center + sigma * rng.normal(size=(n_samples, dim))
        x_new = reflect_bounds(x_new, bounds)

        f_new = evaluator.eval_pop(x_new)
        x[order] = x_new
        fit[order] = f_new

        improved = f_new < pbest_fit[order]
        if np.any(improved):
            improved_idx = order[improved]
            pbest[improved_idx] = x_new[improved]
            pbest_fit[improved_idx] = f_new[improved]

            g_idx = int(np.argmin(pbest_fit))
            if pbest_fit[g_idx] < best:
                best = float(pbest_fit[g_idx])
                gbest = pbest[g_idx].copy()

        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="BBPSO")

    return {"best": best, "fes": evaluator.fes, "curve": recorder.as_dict()}


def qpso_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    x = uniform_init(pop_size, dim, bounds, rng)
    fit = evaluator.eval_pop(x)
    pbest = x.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(pbest_fit))
    gbest = pbest[g_idx].copy()
    best = float(pbest_fit[g_idx])



    beta_max = 1.0
    beta_min = 0.5

    while evaluator.fes < max_fes:
        n_samples = min(pop_size, evaluator.remaining())
        if n_samples <= 0:
            break

        progress = evaluator.fes / max_fes
        beta = beta_max - (beta_max - beta_min) * progress
        mbest = np.mean(pbest, axis=0)

        order = rng.permutation(pop_size)[:n_samples]
        phi = rng.random((n_samples, dim))
        p_local = phi * pbest[order] + (1.0 - phi) * gbest[None, :]
        u = np.maximum(rng.random((n_samples, dim)), 1e-300)
        direction = np.where(rng.random((n_samples, dim)) < 0.5, -1.0, 1.0)
        step = beta * np.abs(mbest[None, :] - x[order]) * np.log(1.0 / u)
        x_new = p_local + direction * step
        x_new = reflect_bounds(x_new, bounds)

        f_new = evaluator.eval_pop(x_new)
        x[order] = x_new
        fit[order] = f_new

        improved = f_new < pbest_fit[order]
        if np.any(improved):
            improved_idx = order[improved]
            pbest[improved_idx] = x_new[improved]
            pbest_fit[improved_idx] = f_new[improved]

            g_idx = int(np.argmin(pbest_fit))
            if pbest_fit[g_idx] < best:
                best = float(pbest_fit[g_idx])
                gbest = pbest[g_idx].copy()

        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="QPSO")

    return {"best": best, "fes": evaluator.fes, "curve": recorder.as_dict()}


def fdr_pso_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    x = uniform_init(pop_size, dim, bounds, rng)
    v = np.zeros_like(x)
    fit = evaluator.eval_pop(x)
    pbest = x.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(pbest_fit))
    gbest = pbest[g_idx].copy()
    best = float(pbest_fit[g_idx])

    c1 = 2.0
    c2 = 2.0
    c3 = 2.0
    eps = 1e-12

    while evaluator.fes < max_fes:
        progress = evaluator.fes / max_fes
        w = 0.9 - 0.5 * progress



        nbest = np.empty_like(x)
        for i in range(pop_size):
            better = pbest_fit < pbest_fit[i]
            better[i] = False
            better_indices = np.where(better)[0]
            if better_indices.size == 0:
                nbest[i] = gbest
                continue

            fitness_gain = pbest_fit[i] - pbest_fit[better_indices]
            for d in range(dim):
                dist = np.abs(pbest[better_indices, d] - x[i, d]) + eps
                ratios = fitness_gain / dist
                j = better_indices[int(np.argmax(ratios))]
                nbest[i, d] = pbest[j, d]

        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        r3 = rng.random((pop_size, dim))
        v = (
            w * v
            + c1 * r1 * (pbest - x)
            + c2 * r2 * (gbest[None, :] - x)
            + c3 * r3 * (nbest - x)
        )
        v = clamp_velocity(v, bounds, 0.2)
        x = reflect_bounds(x + v, bounds)

        fit = evaluator.eval_pop(x)
        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]

        g_idx = int(np.argmin(pbest_fit))
        if pbest_fit[g_idx] < best:
            best = float(pbest_fit[g_idx])
            gbest = pbest[g_idx].copy()

        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="FDR_PSO")

    return {"best": best, "fes": evaluator.fes, "curve": recorder.as_dict()}
def cma_es_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    lb, ub = bounds
    search_range = ub - lb


    lam = int(pop_size)
    mu = max(1, lam // 2)

    weights = np.array(
        [np.log((lam + 1.0) / 2.0) - np.log(i + 1.0) for i in range(mu)],
        dtype=np.float64
    )
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)


    cc = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
    cs = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
    cmu = min(
        1.0 - c1,
        2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff)
    )
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + cs

    mean = rng.uniform(lb, ub, size=dim).astype(np.float64)
    sigma = 0.3 * search_range


    sigma_min = 1e-12 * search_range
    sigma_max = 10.0 * search_range

    C = np.eye(dim, dtype=np.float64)
    pc = np.zeros(dim, dtype=np.float64)
    ps = np.zeros(dim, dtype=np.float64)
    B = np.eye(dim, dtype=np.float64)
    D = np.ones(dim, dtype=np.float64)
    invsqrtC = np.eye(dim, dtype=np.float64)

    chiN = math.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

    best = np.inf
    eigeneval = 0

    def sanitize_covariance():
        nonlocal C, B, D, invsqrtC, pc, ps
        C = np.asarray(C, dtype=np.float64)
        C = np.triu(C) + np.triu(C, 1).T

        if not np.all(np.isfinite(C)):
            C = np.eye(dim, dtype=np.float64)
            B = np.eye(dim, dtype=np.float64)
            D = np.ones(dim, dtype=np.float64)
            invsqrtC = np.eye(dim, dtype=np.float64)
            pc = np.zeros(dim, dtype=np.float64)
            ps = np.zeros(dim, dtype=np.float64)
            return

        try:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.asarray(eigvals, dtype=np.float64)

            if not np.all(np.isfinite(eigvals)):
                raise np.linalg.LinAlgError("non-finite eigenvalues")


            eigvals = np.clip(eigvals, 1e-30, 1e30)

            D_local = np.sqrt(eigvals)
            B_local = eigvecs
            invsqrt_local = B_local @ np.diag(1.0 / D_local) @ B_local.T

            if not np.all(np.isfinite(invsqrt_local)):
                raise np.linalg.LinAlgError("non-finite invsqrtC")

            B = B_local
            D = D_local
            invsqrtC = invsqrt_local

        except np.linalg.LinAlgError:
            C = np.eye(dim, dtype=np.float64)
            B = np.eye(dim, dtype=np.float64)
            D = np.ones(dim, dtype=np.float64)
            invsqrtC = np.eye(dim, dtype=np.float64)
            pc = np.zeros(dim, dtype=np.float64)
            ps = np.zeros(dim, dtype=np.float64)

    while evaluator.fes < max_fes:

        if evaluator.fes - eigeneval > max(1.0, lam / max(c1 + cmu, 1e-12) / dim / 10.0):
            eigeneval = evaluator.fes
            sanitize_covariance()

        n_samples = min(lam, evaluator.remaining())
        if n_samples <= 0:
            break


        z = rng.normal(size=(n_samples, dim))
        y = z @ (B * D).T


        if not np.all(np.isfinite(y)):
            C = np.eye(dim, dtype=np.float64)
            B = np.eye(dim, dtype=np.float64)
            D = np.ones(dim, dtype=np.float64)
            invsqrtC = np.eye(dim, dtype=np.float64)
            pc = np.zeros(dim, dtype=np.float64)
            ps = np.zeros(dim, dtype=np.float64)
            sigma = min(max(sigma, sigma_min), sigma_max)
            y = z

        x = mean[None, :] + sigma * y
        x = reflect_bounds(x, bounds)

        fit = evaluator.eval_pop(x)
        order = np.argsort(fit)

        if np.isfinite(fit[order[0]]) and fit[order[0]] < best:
            best = float(fit[order[0]])


        if n_samples < mu:
            recorder.update(evaluator.fes, best, diversity=np.nan, state="CMA_ES")
            continue

        old_mean = mean.copy()
        x_sel = x[order[:mu]]
        y_sel = (x_sel - old_mean[None, :]) / max(sigma, sigma_min)

        mean = np.sum(weights[:, None] * x_sel, axis=0)
        y_w = np.sum(weights[:, None] * y_sel, axis=0)


        ps = (1.0 - cs) * ps + math.sqrt(cs * (2.0 - cs) * mu_eff) * (invsqrtC @ y_w)

        if not np.all(np.isfinite(ps)):
            ps = np.zeros(dim, dtype=np.float64)

        ps_norm = float(np.linalg.norm(ps))
        if not np.isfinite(ps_norm):
            ps_norm = 0.0
            ps = np.zeros(dim, dtype=np.float64)


        denom = math.sqrt(max(1e-30, 1.0 - (1.0 - cs) ** (2.0 * max(1.0, evaluator.fes / max(lam, 1)))))
        hsig = float(ps_norm / denom / chiN < (1.4 + 2.0 / (dim + 1.0)))

        pc = (1.0 - cc) * pc + hsig * math.sqrt(cc * (2.0 - cc) * mu_eff) * y_w
        if not np.all(np.isfinite(pc)):
            pc = np.zeros(dim, dtype=np.float64)


        rank_mu = np.zeros((dim, dim), dtype=np.float64)
        for i in range(mu):
            yi = y_sel[i]
            if np.all(np.isfinite(yi)):
                rank_mu += weights[i] * np.outer(yi, yi)

        C = (
            (1.0 - c1 - cmu) * C
            + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C)
            + cmu * rank_mu
        )


        C = np.triu(C) + np.triu(C, 1).T
        if not np.all(np.isfinite(C)):
            C = np.eye(dim, dtype=np.float64)
            pc = np.zeros(dim, dtype=np.float64)
            ps = np.zeros(dim, dtype=np.float64)


        expo = (cs / damps) * (ps_norm / chiN - 1.0)


        expo = float(np.clip(expo, -20.0, 20.0))

        sigma *= math.exp(expo)
        sigma = float(np.clip(sigma, sigma_min, sigma_max))


        if not np.all(np.isfinite(mean)):
            mean = rng.uniform(lb, ub, size=dim).astype(np.float64)
            sigma = 0.3 * search_range
            C = np.eye(dim, dtype=np.float64)
            pc = np.zeros(dim, dtype=np.float64)
            ps = np.zeros(dim, dtype=np.float64)
            B = np.eye(dim, dtype=np.float64)
            D = np.ones(dim, dtype=np.float64)
            invsqrtC = np.eye(dim, dtype=np.float64)

        recorder.update(evaluator.fes, best, diversity=np.nan, state="CMA_ES")

    if not np.isfinite(best):

        best = 1e300

    return {
        "best": float(best),
        "fes": evaluator.fes,
        "curve": recorder.as_dict(),
    }


def cso_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim=dim, func_num=func_num, max_fes=max_fes, so_path=so_path)
    recorder = CurveRecorder(max_fes=max_fes, n_points=CFG.CURVE_POINTS)




    n = int(max(pop_size, round(1.5 * dim)))
    n = int(min(max(n, 30), 150))
    if n % 2 == 1:
        n += 1
    n = min(n, max(2, max_fes // 10))
    if n % 2 == 1:
        n -= 1
    n = max(2, n)




    if dim <= 100:
        phi = 0.0
    else:
        phi = max(0.0, 0.14 * math.log(max(n, 2)) - 0.30)

    x = uniform_init(n, dim, bounds, rng)
    v = np.zeros_like(x)
    fit = evaluator.eval_pop(x)
    best_idx = int(np.argmin(fit))
    best = float(fit[best_idx])

    while evaluator.fes < max_fes:



        perm = rng.permutation(n)
        x_old = x.copy()
        v_old = v.copy()
        fit_old = fit.copy()
        mean_x = np.mean(x_old, axis=0)

        new_x = x_old.copy()
        new_v = v_old.copy()
        new_fit = fit_old.copy()

        losers = []
        loser_candidates = []
        loser_velocities = []

        for a in range(0, n, 2):
            i = int(perm[a])
            j = int(perm[a + 1])
            if fit_old[i] <= fit_old[j]:
                w_idx, l_idx = i, j
            else:
                w_idx, l_idx = j, i


            r1 = rng.random(dim)
            r2 = rng.random(dim)
            r3 = rng.random(dim)
            vl = (
                r1 * v_old[l_idx]
                + r2 * (x_old[w_idx] - x_old[l_idx])
                + phi * r3 * (mean_x - x_old[l_idx])
            )
            vl = clamp_velocity(vl, bounds, vmax_ratio=0.2)
            xl = reflect_bounds(x_old[l_idx] + vl, bounds)

            losers.append(l_idx)
            loser_candidates.append(xl)
            loser_velocities.append(vl)

        if not losers:
            break

        loser_candidates_arr = np.asarray(loser_candidates, dtype=np.float64)
        evaluated = evaluator.eval_pop(loser_candidates_arr)

        for pos, l_idx in enumerate(losers):
            if pos < len(evaluated) and np.isfinite(evaluated[pos]):
                new_x[l_idx] = loser_candidates_arr[pos]
                new_v[l_idx] = loser_velocities[pos]
                new_fit[l_idx] = evaluated[pos]

        x, v, fit = new_x, new_v, new_fit
        idx = int(np.argmin(fit))
        if fit[idx] < best:
            best = float(fit[idx])

        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="CSO")

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}

def hpso_gwo_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    x = uniform_init(pop_size, dim, bounds, rng)
    v = np.zeros_like(x)
    fit = evaluator.eval_pop(x)
    best = float(np.min(fit))

    c1 = c2 = c3 = 0.5

    while evaluator.fes < max_fes:
        order = np.argsort(fit)
        alpha = x[order[0]].copy()
        beta = x[order[1]].copy()
        delta = x[order[2]].copy()

        progress = evaluator.fes / max_fes
        l = 2.0 * (1.0 - progress)
        w = 0.5 + rng.random() / 2.0

        new_x = x.copy()
        new_v = v.copy()
        for i in range(pop_size):
            A1 = 2.0 * l * rng.random(dim) - l
            A2 = 2.0 * l * rng.random(dim) - l
            A3 = 2.0 * l * rng.random(dim) - l
            C1 = 2.0 * rng.random(dim)
            C2 = 2.0 * rng.random(dim)
            C3 = 2.0 * rng.random(dim)

            D_alpha = np.abs(C1 * alpha - w * x[i])
            D_beta = np.abs(C2 * beta - w * x[i])
            D_delta = np.abs(C3 * delta - w * x[i])

            X1 = alpha - A1 * D_alpha
            X2 = beta - A2 * D_beta
            X3 = delta - A3 * D_delta

            r1 = rng.random(dim)
            r2 = rng.random(dim)
            r3 = rng.random(dim)
            new_v[i] = w * (v[i] + c1 * r1 * (X1 - x[i]) + c2 * r2 * (X2 - x[i]) + c3 * r3 * (X3 - x[i]))
            new_x[i] = x[i] + new_v[i]

        v = clamp_velocity(new_v, bounds, 0.2)
        x = reflect_bounds(x + v, bounds)
        fit = evaluator.eval_pop(x)
        current = float(np.min(fit))
        if current < best:
            best = current
        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="HPSOGWO")

    return {"best": best, "fes": evaluator.fes, "curve": recorder.as_dict()}



L_SHADE_PARAMS: Dict[str, float] = {


    "R_NINIT": 18.0,
    "R_ARC": 2.6,
    "P": 0.11,
    "H": 6,
    "N_MIN": 4,
}


def _weighted_lehmer_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weights * values))
    if denom <= 1e-300:
        return float(np.mean(values)) if len(values) else 0.5
    return float(np.sum(weights * values * values) / denom)


def _sample_lshade_F(mean_f: float, rng: np.random.Generator) -> float:

    for _ in range(100):
        f = float(mean_f + 0.1 * rng.standard_cauchy())
        if f > 0.0:
            return min(f, 1.0)
    return min(max(float(mean_f), 1e-8), 1.0)


def _shade_boundary_correction(mutant: np.ndarray, parent: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:

    lb, ub = bounds
    v = np.asarray(mutant, dtype=np.float64).copy()
    v = np.where(v < lb, (lb + parent) / 2.0, v)
    v = np.where(v > ub, (ub + parent) / 2.0, v)
    return np.clip(v, lb, ub)


def lshade_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    r_ninit = float(L_SHADE_PARAMS["R_NINIT"])
    r_arc = float(L_SHADE_PARAMS["R_ARC"])
    p_best_rate = float(L_SHADE_PARAMS["P"])
    h_size = int(L_SHADE_PARAMS["H"])
    n_min = int(L_SHADE_PARAMS["N_MIN"])

    n_init = max(n_min, int(round(dim * r_ninit)))
    n_current = n_init


    pop = uniform_init(n_current, dim, bounds, rng)
    fit = evaluator.eval_pop(pop)
    best = float(np.min(fit))


    m_cr = np.full(h_size, 0.5, dtype=np.float64)
    m_f = np.full(h_size, 0.5, dtype=np.float64)
    memory_pos = 0


    archive = np.empty((0, dim), dtype=np.float64)

    while evaluator.fes < max_fes and n_current >= n_min:
        n_current = pop.shape[0]
        if n_current < n_min:
            break

        order = np.argsort(fit)
        p_num = max(2, int(round(p_best_rate * n_current)))
        p_num = min(p_num, n_current)
        top_indices = order[:p_num]

        trial = np.empty_like(pop)
        cr_values = np.empty(n_current, dtype=np.float64)
        f_values = np.empty(n_current, dtype=np.float64)


        union = pop if archive.shape[0] == 0 else np.vstack([pop, archive])
        union_size = union.shape[0]

        for i in range(n_current):
            mem_idx = int(rng.integers(0, h_size))

            if np.isnan(m_cr[mem_idx]):
                cr = 0.0
            else:
                cr = float(rng.normal(m_cr[mem_idx], 0.1))
                cr = float(np.clip(cr, 0.0, 1.0))

            f = _sample_lshade_F(m_f[mem_idx], rng)
            cr_values[i] = cr
            f_values[i] = f

            pbest_idx = int(rng.choice(top_indices))


            candidates_r1 = np.arange(n_current)
            if n_current > 1:
                candidates_r1 = candidates_r1[candidates_r1 != i]
            r1_idx = int(rng.choice(candidates_r1))



            while True:
                r2_idx = int(rng.integers(0, union_size))
                if r2_idx == i:
                    continue
                if r2_idx == r1_idx:
                    continue
                break

            mutant = (
                pop[i]
                + f * (pop[pbest_idx] - pop[i])
                + f * (pop[r1_idx] - union[r2_idx])
            )
            mutant = _shade_boundary_correction(mutant, pop[i], bounds)

            cross = rng.random(dim) <= cr
            cross[int(rng.integers(0, dim))] = True
            trial[i] = np.where(cross, mutant, pop[i])

        trial_fit = evaluator.eval_pop(trial)


        s_cr: List[float] = []
        s_f: List[float] = []
        delta_f: List[float] = []
        new_archive_members: List[np.ndarray] = []


        evaluated_mask = np.isfinite(trial_fit)
        for i in range(n_current):
            if not evaluated_mask[i]:
                continue
            if trial_fit[i] <= fit[i]:
                if trial_fit[i] < fit[i]:
                    new_archive_members.append(pop[i].copy())
                    s_cr.append(float(cr_values[i]))
                    s_f.append(float(f_values[i]))
                    delta_f.append(float(abs(fit[i] - trial_fit[i])))
                pop[i] = trial[i]
                fit[i] = trial_fit[i]

        current_best = float(np.min(fit))
        if current_best < best:
            best = current_best


        if new_archive_members:
            archive = np.vstack([archive, np.asarray(new_archive_members, dtype=np.float64)])

        archive_max = max(0, int(round(r_arc * pop.shape[0])))
        if archive.shape[0] > archive_max:
            keep = rng.choice(archive.shape[0], size=archive_max, replace=False)
            archive = archive[keep]


        if len(s_f) > 0 and len(s_cr) > 0:
            s_cr_arr = np.asarray(s_cr, dtype=np.float64)
            s_f_arr = np.asarray(s_f, dtype=np.float64)
            df_arr = np.asarray(delta_f, dtype=np.float64)
            if float(np.sum(df_arr)) <= 0.0:
                weights = np.full_like(df_arr, 1.0 / len(df_arr))
            else:
                weights = df_arr / np.sum(df_arr)

            if np.isnan(m_cr[memory_pos]) or np.max(s_cr_arr) == 0.0:
                m_cr[memory_pos] = np.nan
            else:
                m_cr[memory_pos] = _weighted_lehmer_mean(s_cr_arr, weights)
            m_f[memory_pos] = _weighted_lehmer_mean(s_f_arr, weights)
            memory_pos = (memory_pos + 1) % h_size


        target_n = int(round(((n_min - n_init) / float(max_fes)) * evaluator.fes + n_init))
        target_n = int(np.clip(target_n, n_min, n_init))
        if target_n < pop.shape[0]:
            order = np.argsort(fit)
            keep = order[:target_n]
            pop = pop[keep]
            fit = fit[keep]

            archive_max = max(0, int(round(r_arc * pop.shape[0])))
            if archive.shape[0] > archive_max:
                keep_a = rng.choice(archive.shape[0], size=archive_max, replace=False)
                archive = archive[keep_a]

        recorder.update(evaluator.fes, best, diversity=population_diversity(pop, bounds), state="L_SHADE")

        if evaluator.remaining() <= 0:
            break

    return {"best": best, "fes": evaluator.fes, "curve": recorder.as_dict()}


J_SO_PARAMS: Dict[str, float] = {







    "P_MAX": 0.25,
    "P_MIN_RATIO": 0.5,
    "H": 5,
    "R_ARC": 2.6,
    "N_MIN": 4,
    "MF_INIT": 0.3,
    "MCR_INIT": 0.8,
}


def _jso_weighted_fw(f: float, nfes: int, max_fes: int) -> float:

    progress = nfes / max(1, max_fes)
    if progress < 0.2:
        return 0.7 * f
    if progress < 0.4:
        return 0.8 * f
    return 1.2 * f


def _jso_population_size(dim: int, max_fes: int, n_min: int) -> int:

    n_init = int(round(25.0 * math.log(max(2.0, float(dim))) * math.sqrt(float(dim))))
    n_init = max(n_min, n_init)

    n_init = min(n_init, max(n_min, max_fes // 10))
    return int(n_init)


def jso_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    p_max = float(J_SO_PARAMS["P_MAX"])
    p_min = p_max * float(J_SO_PARAMS["P_MIN_RATIO"])
    h_size = int(J_SO_PARAMS["H"])
    r_arc = float(J_SO_PARAMS["R_ARC"])
    n_min = int(J_SO_PARAMS["N_MIN"])

    n_init = _jso_population_size(dim, max_fes, n_min)
    pop = uniform_init(n_init, dim, bounds, rng)
    fit = evaluator.eval_pop(pop)
    best = float(np.min(fit))


    m_cr = np.full(h_size, float(J_SO_PARAMS["MCR_INIT"]), dtype=np.float64)
    m_f = np.full(h_size, float(J_SO_PARAMS["MF_INIT"]), dtype=np.float64)
    memory_pos = 0
    archive = np.empty((0, dim), dtype=np.float64)

    while evaluator.fes < max_fes and pop.shape[0] >= n_min:
        n_current = pop.shape[0]
        progress = evaluator.fes / max(1, max_fes)


        p_current = p_max - (p_max - p_min) * progress
        p_current = float(np.clip(p_current, p_min, p_max))

        order = np.argsort(fit)
        p_num = max(2, int(round(p_current * n_current)))
        p_num = min(p_num, n_current)
        top_indices = order[:p_num]

        union = pop if archive.shape[0] == 0 else np.vstack([pop, archive])
        union_size = union.shape[0]

        trial = np.empty_like(pop)
        cr_values = np.empty(n_current, dtype=np.float64)
        f_values = np.empty(n_current, dtype=np.float64)

        for i in range(n_current):
            mem_idx = int(rng.integers(0, h_size))


            if mem_idx == h_size - 1:
                mean_cr = 0.9
                mean_f = 0.9
            else:
                mean_cr = m_cr[mem_idx]
                mean_f = m_f[mem_idx]

            if np.isnan(mean_cr) or mean_cr < 0.0:
                cr = 0.0
            else:
                cr = float(np.clip(rng.normal(mean_cr, 0.1), 0.0, 1.0))


            progress_now = evaluator.fes / max(1, max_fes)
            if progress_now < 0.25:
                cr = max(cr, 0.7)
            elif progress_now < 0.50:
                cr = max(cr, 0.6)

            f = _sample_lshade_F(mean_f, rng)
            if progress_now < 0.60 and f > 0.7:
                f = 0.7

            cr_values[i] = cr
            f_values[i] = f
            fw = _jso_weighted_fw(f, evaluator.fes, max_fes)

            pbest_idx = int(rng.choice(top_indices))

            candidates_r1 = np.arange(n_current)
            candidates_r1 = candidates_r1[candidates_r1 != i]
            r1_idx = int(rng.choice(candidates_r1))

            while True:
                r2_idx = int(rng.integers(0, union_size))
                if r2_idx == i:
                    continue
                if r2_idx == r1_idx:
                    continue
                break

            mutant = (
                pop[i]
                + fw * (pop[pbest_idx] - pop[i])
                + f * (pop[r1_idx] - union[r2_idx])
            )
            mutant = _shade_boundary_correction(mutant, pop[i], bounds)

            cross = rng.random(dim) <= cr
            cross[int(rng.integers(0, dim))] = True
            trial[i] = np.where(cross, mutant, pop[i])

        trial_fit = evaluator.eval_pop(trial)

        s_cr: List[float] = []
        s_f: List[float] = []
        delta_f: List[float] = []
        new_archive_members: List[np.ndarray] = []

        evaluated_mask = np.isfinite(trial_fit)
        for i in range(n_current):
            if not evaluated_mask[i]:
                continue
            if trial_fit[i] <= fit[i]:
                if trial_fit[i] < fit[i]:
                    new_archive_members.append(pop[i].copy())
                    s_cr.append(float(cr_values[i]))
                    s_f.append(float(f_values[i]))
                    delta_f.append(float(abs(fit[i] - trial_fit[i])))
                pop[i] = trial[i]
                fit[i] = trial_fit[i]

        current_best = float(np.min(fit))
        if current_best < best:
            best = current_best

        if new_archive_members:
            archive = np.vstack([archive, np.asarray(new_archive_members, dtype=np.float64)])

        archive_max = max(0, int(round(r_arc * pop.shape[0])))
        if archive.shape[0] > archive_max:
            keep = rng.choice(archive.shape[0], size=archive_max, replace=False)
            archive = archive[keep]


        if len(s_f) > 0 and len(s_cr) > 0:
            s_cr_arr = np.asarray(s_cr, dtype=np.float64)
            s_f_arr = np.asarray(s_f, dtype=np.float64)
            df_arr = np.asarray(delta_f, dtype=np.float64)
            if float(np.sum(df_arr)) <= 0.0:
                weights = np.full_like(df_arr, 1.0 / len(df_arr))
            else:
                weights = df_arr / np.sum(df_arr)

            if np.isnan(m_cr[memory_pos]) or np.max(s_cr_arr) == 0.0:
                new_mcr = np.nan
            else:
                new_mcr = _weighted_lehmer_mean(s_cr_arr, weights)
                if not np.isnan(m_cr[memory_pos]):
                    new_mcr = 0.5 * m_cr[memory_pos] + 0.5 * new_mcr

            new_mf = _weighted_lehmer_mean(s_f_arr, weights)
            new_mf = 0.5 * m_f[memory_pos] + 0.5 * new_mf


            m_cr[memory_pos] = new_mcr
            m_f[memory_pos] = new_mf
            memory_pos = (memory_pos + 1) % h_size


        target_n = int(round(((n_min - n_init) / float(max_fes)) * evaluator.fes + n_init))
        target_n = int(np.clip(target_n, n_min, n_init))
        if target_n < pop.shape[0]:
            keep = np.argsort(fit)[:target_n]
            pop = pop[keep]
            fit = fit[keep]
            archive_max = max(0, int(round(r_arc * pop.shape[0])))
            if archive.shape[0] > archive_max:
                keep_a = rng.choice(archive.shape[0], size=archive_max, replace=False)
                archive = archive[keep_a]

        recorder.update(evaluator.fes, best, diversity=population_diversity(pop, bounds), state="JSO")
        if evaluator.remaining() <= 0:
            break

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}




def dms_pso_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    n = int(pop_size)
    m = 3
    regroup_period = 5
    c1 = 2.0
    c2 = 2.0

    x = uniform_init(n, dim, bounds, rng)
    v = rng.uniform(-0.05 * (bounds[1] - bounds[0]), 0.05 * (bounds[1] - bounds[0]), size=(n, dim))
    fit = evaluator.eval_pop(x)

    pbest = x.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(pbest_fit))
    gbest = pbest[g_idx].copy()
    best = float(pbest_fit[g_idx])


    permutation = rng.permutation(n)
    iteration = 0

    def regroup() -> List[np.ndarray]:
        perm = rng.permutation(n)
        return [perm[i:i + m] for i in range(0, n, m)]

    groups = regroup()

    while evaluator.fes < max_fes:
        progress = evaluator.fes / max_fes
        w = 0.9 - 0.7 * progress



        use_global_phase = progress >= 0.90

        if (not use_global_phase) and (iteration % regroup_period == 0):
            groups = regroup()

        if use_global_phase:
            lbest_matrix = np.tile(gbest, (n, 1))
        else:
            lbest_matrix = np.empty_like(x)
            for group in groups:
                if len(group) == 0:
                    continue
                local_best_idx = int(group[np.argmin(pbest_fit[group])])
                lbest_matrix[group] = pbest[local_best_idx]

        r1 = rng.random((n, dim))
        r2 = rng.random((n, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (lbest_matrix - x)
        v = clamp_velocity(v, bounds, 0.2)
        x = reflect_bounds(x + v, bounds)

        fit = evaluator.eval_pop(x)
        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]

        g_idx = int(np.argmin(pbest_fit))
        if pbest_fit[g_idx] < best:
            best = float(pbest_fit[g_idx])
            gbest = pbest[g_idx].copy()

        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="DMS_PSO-global" if use_global_phase else "DMS_PSO-local")
        iteration += 1

    return {"best": best, "fes": evaluator.fes, "curve": recorder.as_dict()}


def sl_pso_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim, func_num, max_fes, so_path)
    recorder = CurveRecorder(max_fes, CFG.CURVE_POINTS)

    M = 100
    alpha = 0.5
    beta = 0.01
    swarm_size = int(M + dim // 10)

    swarm_size = min(swarm_size, max(2, max_fes // 10))
    epsilon = beta * dim / float(M)

    x = uniform_init(swarm_size, dim, bounds, rng)
    delta_x = np.zeros_like(x)
    fit = evaluator.eval_pop(x)

    best_idx = int(np.argmin(fit))
    best = float(fit[best_idx])
    best_x = x[best_idx].copy()

    ceil_dim_over_M = max(1, int(math.ceil(dim / float(M))))
    exponent = alpha * math.log(ceil_dim_over_M)



    while evaluator.fes < max_fes:

        order = np.argsort(fit)[::-1]
        x = x[order]
        fit = fit[order]
        delta_x = delta_x[order]

        current_best_idx = swarm_size - 1
        mean_behavior = np.mean(x, axis=0)
        new_x = x.copy()
        new_delta = delta_x.copy()

        for i in range(swarm_size - 1):

            p_learning = (1.0 - (i / float(swarm_size))) ** exponent if exponent > 0.0 else 1.0
            p_learning = float(np.clip(p_learning, 0.0, 1.0))
            if rng.random() <= p_learning:


                demonstrators = rng.integers(i + 1, swarm_size, size=dim)
                demo_values = x[demonstrators, np.arange(dim)]

                r1 = rng.random(dim)
                r2 = rng.random(dim)
                r3 = rng.random(dim)
                imitation = demo_values - x[i]
                social_influence = mean_behavior - x[i]
                new_delta[i] = r1 * delta_x[i] + r2 * imitation + r3 * epsilon * social_influence
                new_x[i] = x[i] + new_delta[i]


        new_x[current_best_idx] = x[current_best_idx]
        new_delta[current_best_idx] = delta_x[current_best_idx]

        x = reflect_bounds(new_x, bounds)
        delta_x = clamp_velocity(new_delta, bounds, 0.2)
        fit = evaluator.eval_pop(x)

        idx = int(np.argmin(fit))
        if fit[idx] < best:
            best = float(fit[idx])
            best_x = x[idx].copy()

        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="SL_PSO")

    return {"best": best, "fes": evaluator.fes, "curve": recorder.as_dict()}


# Additional PSO variants
def _bounded_archive_append(
    archive_x: np.ndarray,
    archive_f: np.ndarray,
    x_new: np.ndarray,
    f_new: float,
    max_len: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:

    x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
    f_new = float(f_new)
    if archive_x.size == 0:
        return x_new.copy(), np.array([f_new], dtype=np.float64)
    if archive_x.shape[0] < max_len:
        return np.vstack([archive_x, x_new]), np.append(archive_f, f_new)
    if max_len <= 0:
        return archive_x, archive_f
    if archive_x.shape[0] == 1:
        if f_new < archive_f[0]:
            archive_x[0] = x_new[0]
            archive_f[0] = f_new
        return archive_x, archive_f
    a, b = rng.choice(archive_x.shape[0], size=2, replace=False)
    worse = int(a if archive_f[a] >= archive_f[b] else b)
    if f_new < archive_f[worse]:
        archive_x[worse] = x_new[0]
        archive_f[worse] = f_new
    return archive_x, archive_f


def eapso_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim=dim, func_num=func_num, max_fes=max_fes, so_path=so_path)
    recorder = CurveRecorder(max_fes=max_fes, n_points=CFG.CURVE_POINTS)

    n = int(pop_size)
    if n % 2 == 1:
        n += 1
    n = min(n, max(2, max_fes // 10))
    if n % 2 == 1:
        n -= 1
    n = max(2, n)

    x = uniform_init(n, dim, bounds, rng)
    v = np.zeros_like(x)
    fit = evaluator.eval_pop(x)
    pbest = x.copy()
    pbest_fit = fit.copy()
    best_idx = int(np.argmin(pbest_fit))
    gbest = pbest[best_idx].copy()
    best = float(pbest_fit[best_idx])

    archive_b_x = np.empty((0, dim), dtype=np.float64)
    archive_b_f = np.empty(0, dtype=np.float64)
    archive_c_x = gbest.reshape(1, -1).copy()
    archive_c_f = np.array([best], dtype=np.float64)

    while evaluator.fes < max_fes:
        order = np.argsort(fit)
        half = n // 2
        outstanding = order[:half]
        common = order[half:]


        archive_a_x = pbest[outstanding].copy()
        archive_a_f = pbest_fit[outstanding].copy()


        if archive_b_x.shape[0] == 0:
            archive_b_x = pbest[order[:min(n, max(1, half))]].copy()
            archive_b_f = pbest_fit[order[:min(n, max(1, half))]].copy()
        if archive_c_x.shape[0] == 0:
            archive_c_x = gbest.reshape(1, -1).copy()
            archive_c_f = np.array([best], dtype=np.float64)

        mean_common_fit = float(np.mean(fit[common])) if common.size else float(np.mean(fit))
        trial_x = []
        trial_idx = []
        trial_v = []

        for idx in common:
            if len(trial_x) >= evaluator.remaining():
                break

            a_i = int(rng.integers(0, archive_a_x.shape[0]))
            b_i = int(rng.integers(0, archive_b_x.shape[0]))
            c_i = int(rng.integers(0, archive_c_x.shape[0]))

            a_x, b_x, c_x = archive_a_x[a_i], archive_b_x[b_i], archive_c_x[c_i]
            a_f, b_f, c_f = archive_a_f[a_i], archive_b_f[b_i], archive_c_f[c_i]


            w_r = rng.random(dim)
            lam1 = rng.random(dim)
            lam2 = rng.random(dim)

            if fit[idx] < mean_common_fit:

                if a_f <= b_f and a_f <= c_f:
                    e1, e2 = a_x, gbest
                elif b_f <= a_f and b_f <= c_f:
                    e1, e2 = b_x, gbest
                else:
                    e1, e2 = c_x, gbest
            else:

                samples = [(a_f, a_x), (b_f, b_x), (c_f, c_x)]
                samples.sort(key=lambda z: z[0])
                e1, e2 = samples[0][1], samples[1][1]

            v_new = w_r * v[idx] + lam1 * (e1 - x[idx]) + lam2 * (e2 - x[idx])
            v_new = clamp_velocity(v_new, bounds, 0.2)
            x_new = reflect_bounds(x[idx] + v_new, bounds)
            trial_x.append(x_new)
            trial_idx.append(int(idx))
            trial_v.append(v_new)

        if not trial_x:
            break

        trials = np.asarray(trial_x, dtype=np.float64)
        trial_fit = evaluator.eval_pop(trials)

        for k, idx in enumerate(trial_idx):
            if not np.isfinite(trial_fit[k]):
                continue
            x[idx] = trials[k]
            v[idx] = trial_v[k]
            fit[idx] = trial_fit[k]

            if trial_fit[k] < pbest_fit[idx]:
                pbest[idx] = trials[k]
                pbest_fit[idx] = trial_fit[k]
                archive_b_x, archive_b_f = _bounded_archive_append(
                    archive_b_x, archive_b_f, trials[k], float(trial_fit[k]), n, rng
                )
                if trial_fit[k] < best:
                    best = float(trial_fit[k])
                    gbest = trials[k].copy()
                    archive_c_x, archive_c_f = _bounded_archive_append(
                        archive_c_x, archive_c_f, gbest, best, n, rng
                    )


        archive_c_x, archive_c_f = _bounded_archive_append(archive_c_x, archive_c_f, gbest, best, n, rng)
        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="EAPSO")

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}


def _dbcd_entropy_local_diversity(pop: np.ndarray) -> np.ndarray:


    n = pop.shape[0]
    eld = np.zeros(n, dtype=np.float64)
    if n <= 2:
        return eld

    d1 = np.linalg.norm(pop[1:-1] - pop[:-2], axis=1)
    d2 = np.linalg.norm(pop[2:] - pop[1:-1], axis=1)
    Dsum = d1 + d2

    ns = Dsum.copy()
    p1 = np.divide(d1, Dsum, out=np.zeros_like(d1), where=Dsum > 1e-300)
    p2 = np.divide(d2, Dsum, out=np.zeros_like(d2), where=Dsum > 1e-300)
    nd = np.zeros_like(d1)
    mask1 = p1 > 1e-300
    mask2 = p2 > 1e-300
    nd[mask1] -= p1[mask1] * np.log(p1[mask1])
    nd[mask2] -= p2[mask2] * np.log(p2[mask2])

    def norm01(a: np.ndarray) -> np.ndarray:
        if a.size == 0:
            return a
        lo, hi = float(np.min(a)), float(np.max(a))
        if hi - lo <= 1e-300:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    eld[1:-1] = norm01(ns) * norm01(nd)
    return eld


def pso_dbcd_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim=dim, func_num=func_num, max_fes=max_fes, so_path=so_path)
    recorder = CurveRecorder(max_fes=max_fes, n_points=CFG.CURVE_POINTS)

    n = int(pop_size)
    if n % 2 == 1:
        n += 1
    n = max(4, n)
    phi = 0.2

    ssw_base = [2, 4, 8, 10, 20, 25, 40, 50]
    ssw = [s for s in ssw_base if s <= n and s >= 2]
    if not ssw:
        ssw = [2]

    x = uniform_init(n, dim, bounds, rng)
    v = np.zeros_like(x)
    fit = evaluator.eval_pop(x)
    best_idx = int(np.argmin(fit))
    gbest = x[best_idx].copy()
    best = float(fit[best_idx])

    while evaluator.fes < max_fes:
        progress = evaluator.fes / max_fes
        stage = min(len(ssw) - 1, int(progress * len(ssw)))
        s = int(ssw[stage])


        perm = rng.permutation(n)
        puc = set()
        pec: Dict[int, int] = {}
        subgroup_worst = set()
        for start in range(0, n, s):
            group = perm[start:start + s]
            if group.size == 0:
                continue
            gbest_local = int(group[np.argmin(fit[group])])
            gworst_local = int(group[np.argmax(fit[group])])
            subgroup_worst.add(gworst_local)
            for idx in group:
                idx = int(idx)
                if idx != gbest_local:
                    puc.add(idx)
                    pec[idx] = gbest_local

        losers = set()
        pair_perm = rng.permutation(n)
        for k in range(0, n - 1, 2):
            i, j = int(pair_perm[k]), int(pair_perm[k + 1])
            losers.add(i if fit[i] > fit[j] else j)
        puc = puc.intersection(losers)


        fit_order = np.argsort(fit)
        x_sorted = x[fit_order]
        eld_sorted = _dbcd_entropy_local_diversity(x_sorted)
        eld = np.zeros(n, dtype=np.float64)
        eld[fit_order] = eld_sorted

        pud = set()
        ped: Dict[int, int] = {}
        for i in range(n):
            candidates = np.where(eld > eld[i] + 1e-300)[0]
            if candidates.size == 0:
                continue
            j = int(rng.choice(candidates))
            if eld[i] <= rng.random() * eld[j] or i in subgroup_worst:
                pud.add(i)
                ped[i] = j

        update_indices = list(puc.intersection(pud))
        if len(update_indices) == 0:

            update_indices = list(losers)[:max(1, min(len(losers), evaluator.remaining()))]
            for i in update_indices:
                if i not in pec:
                    pec[i] = int(np.argmin(fit))
                if i not in ped:

                    ped[i] = int(np.argmax(eld)) if np.max(eld) > 0 else int(np.argmin(fit))

        update_indices = update_indices[:evaluator.remaining()]
        if not update_indices:
            break

        trials = []
        new_vs = []
        for i in update_indices:
            ec = pec.get(i, int(np.argmin(fit)))
            ed = ped.get(i, int(np.argmax(eld)) if np.max(eld) > 0 else int(np.argmin(fit)))
            w_rand = rng.random(dim)
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            v_new = w_rand * v[i] + r1 * (x[ec] - x[i]) + phi * r2 * (x[ed] - x[i])
            v_new = clamp_velocity(v_new, bounds, 0.2)
            x_new = reflect_bounds(x[i] + v_new, bounds)
            trials.append(x_new)
            new_vs.append(v_new)

        trial_fit = evaluator.eval_pop(np.asarray(trials, dtype=np.float64))
        for k, i in enumerate(update_indices):
            if not np.isfinite(trial_fit[k]):
                continue
            x[i] = trials[k]
            v[i] = new_vs[k]
            fit[i] = trial_fit[k]
            if trial_fit[k] < best:
                best = float(trial_fit[k])
                gbest = x[i].copy()

        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="PSO_DBCD")

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}


def _oa_two_level(n_factors: int) -> np.ndarray:

    m = 1
    k = 0
    while m < n_factors + 1:
        k += 1
        m = 2 ** k
    rows = np.arange(m, dtype=np.int64)
    columns = []
    for col_id in range(1, m):
        bit = np.zeros(m, dtype=np.int8)
        for b in range(k):
            if (col_id >> b) & 1:
                bit ^= ((rows >> b) & 1).astype(np.int8)
        columns.append(bit)
        if len(columns) >= n_factors:
            break
    return np.stack(columns, axis=1)


def _construct_ol_exemplar(
    particle_index: int,
    pbest: np.ndarray,
    pbest_fit: np.ndarray,
    lbest: np.ndarray,
    evaluator: BudgetedCEC17,
    bounds: Tuple[float, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, np.ndarray, float]:


    dim = pbest.shape[1]
    pi = pbest[particle_index]
    pn = lbest[particle_index]
    if np.allclose(pi, pn):
        choices = [j for j in range(pbest.shape[0]) if j != particle_index]
        if choices:
            pn = pbest[int(rng.choice(choices))]

    oa = _oa_two_level(dim)
    m = oa.shape[0]
    if evaluator.remaining() <= 0:
        return np.zeros(dim, dtype=np.int8), np.inf, pi.copy(), np.inf
    allowed = min(m, evaluator.remaining())
    test_x = np.where(oa[:allowed].astype(bool), pn[None, :], pi[None, :])
    test_x = reflect_bounds(test_x, bounds)
    test_f = evaluator.eval_pop(test_x)

    best_row = int(np.argmin(test_f))
    best_mask = oa[best_row].copy()
    best_x = test_x[best_row].copy()
    best_f = float(test_f[best_row])


    pred_mask = np.zeros(dim, dtype=np.int8)
    for d in range(dim):
        vals0 = test_f[oa[:allowed, d] == 0]
        vals1 = test_f[oa[:allowed, d] == 1]
        mean0 = float(np.mean(vals0)) if vals0.size else np.inf
        mean1 = float(np.mean(vals1)) if vals1.size else np.inf
        pred_mask[d] = 1 if mean1 < mean0 else 0

    if evaluator.can_eval(1):
        pred_x = np.where(pred_mask.astype(bool), pn, pi)
        pred_x = reflect_bounds(pred_x, bounds)
        pred_f = evaluator.eval_one(pred_x)
        if pred_f < best_f:
            best_mask = pred_mask.copy()
            best_x = pred_x.copy()
            best_f = float(pred_f)

    return best_mask.astype(np.int8), best_f, best_x, best_f


def olpso_algorithm(dim, func_num, pop_size, max_fes, bounds, so_path, seed) -> Dict[str, Any]:


    rng = np.random.default_rng(seed)
    evaluator = BudgetedCEC17(dim=dim, func_num=func_num, max_fes=max_fes, so_path=so_path)
    recorder = CurveRecorder(max_fes=max_fes, n_points=CFG.CURVE_POINTS)

    n = int(pop_size)
    x = uniform_init(n, dim, bounds, rng)
    v = np.zeros_like(x)
    fit = evaluator.eval_pop(x)
    pbest = x.copy()
    pbest_fit = fit.copy()
    best_idx = int(np.argmin(pbest_fit))
    best = float(pbest_fit[best_idx])
    gbest = pbest[best_idx].copy()

    gap = 5
    no_improve = np.full(n, gap, dtype=np.int64)
    masks = np.zeros((n, dim), dtype=np.int8)

    def current_lbest() -> np.ndarray:

        return np.tile(gbest.reshape(1, -1), (n, 1))


    while evaluator.fes < max_fes:
        lbest = current_lbest()


        for i in range(n):
            if evaluator.remaining() <= 0:
                break
            if no_improve[i] >= gap:
                mask, cand_f, cand_x, _ = _construct_ol_exemplar(i, pbest, pbest_fit, lbest, evaluator, bounds, rng)
                masks[i] = mask
                no_improve[i] = 0

                if np.isfinite(cand_f) and cand_f < pbest_fit[i]:
                    pbest[i] = cand_x.copy()
                    pbest_fit[i] = cand_f
                    if cand_f < best:
                        best = float(cand_f)
                        gbest = cand_x.copy()

        if evaluator.remaining() <= 0:
            break

        lbest = current_lbest()
        exemplar = np.where(masks.astype(bool), lbest, pbest)
        progress = evaluator.fes / max_fes
        w = 0.9 - 0.5 * progress
        c = 2.0
        v = w * v + c * rng.random((n, dim)) * (exemplar - x)
        v = clamp_velocity(v, bounds, 0.2)
        x = reflect_bounds(x + v, bounds)
        fit = evaluator.eval_pop(x)

        improved = fit < pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        no_improve[improved] = 0
        no_improve[~improved] += 1

        best_idx = int(np.argmin(pbest_fit))
        if pbest_fit[best_idx] < best:
            best = float(pbest_fit[best_idx])
            gbest = pbest[best_idx].copy()

        recorder.update(evaluator.fes, best, diversity=population_diversity(x, bounds), state="OLPSO")

    return {"best": float(best), "fes": evaluator.fes, "curve": recorder.as_dict()}

ALGORITHM_REGISTRY = {
    "SCLPSO": sclpso_revised,
    "PSO": pso_algorithm,
    "APSO": apso_algorithm,
    "Classic_CLPSO": clpso_algorithm,
    "FIPS": fips_algorithm,
    "DMS_PSO": dms_pso_algorithm,
    "QPSO": qpso_algorithm,
    "FDR_PSO": fdr_pso_algorithm,
    "EAPSO": eapso_algorithm,
    "PSO_DBCD": pso_dbcd_algorithm,
    "OLPSO": olpso_algorithm,
}


# Experiment runner
def run_one_task(args: Tuple[int, int, str, int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dim, func_num, algo, run_idx = args
    max_fes = CFG.MAX_FES_FACTOR * dim
    seed = make_seed(dim, func_num, algo, run_idx)
    bounds = bounds_tuple()

    start = time.time()
    try:
        if algo not in ALGORITHM_REGISTRY:
            raise ValueError(f"Unknown algorithm: {algo}")

        result = ALGORITHM_REGISTRY[algo](
            dim=dim,
            func_num=func_num,
            pop_size=CFG.POP_SIZE,
            max_fes=max_fes,
            bounds=bounds,
            so_path=CFG.SO_PATH,
            seed=seed,
        )

        elapsed = time.time() - start
        row = {
            "dimension": dim,
            "function": func_num,
            "algorithm": algo,
            "run": run_idx + 1,
            "seed": seed,
            "best": float(result["best"]),
            "fes": int(result["fes"]),
            "max_fes": int(max_fes),
            "elapsed_sec": float(elapsed),
            "status": "ok",
            "error": "",
        }
        curve_row = {
            "dimension": dim,
            "function": func_num,
            "algorithm": algo,
            "run": run_idx + 1,
            "seed": seed,
            "curve": result.get("curve", {}),
        }
        if getattr(CFG, "VERBOSE_TASKS", False):
            print(
                f"[OK] {algo:8s} | {dim:3d}D F{func_num:02d} run {run_idx + 1:02d}/{CFG.N_RUNS} "
                f"| best={row['best']:.6e} | FEs={row['fes']} | {elapsed:.1f}s",
                flush=True,
            )
        return row, curve_row

    except Exception as e:
        elapsed = time.time() - start
        err = traceback.format_exc()
        row = {
            "dimension": dim,
            "function": func_num,
            "algorithm": algo,
            "run": run_idx + 1,
            "seed": seed,
            "best": np.inf,
            "fes": 0,
            "max_fes": int(max_fes),
            "elapsed_sec": float(elapsed),
            "status": "failed",
            "error": str(e),
        }
        curve_row = {
            "dimension": dim,
            "function": func_num,
            "algorithm": algo,
            "run": run_idx + 1,
            "seed": seed,
            "curve": {},
        }
        print(f"[FAILED] {algo} | {dim}D F{func_num} run {run_idx + 1}: {e}", flush=True)
        print(err, flush=True)
        return row, curve_row


def build_tasks() -> List[Tuple[int, int, str, int]]:
    if CFG.TEST_MODE:
        dims = (10,)
        funcs = [1, 3]
        algos = ("SCLPSO", "PSO")
        runs = 2
    else:
        dims = CFG.DIMENSIONS
        funcs = make_functions(CFG.EXCLUDE_F2)
        algos = CFG.ALGORITHMS
        runs = CFG.N_RUNS

    tasks = []
    for dim in dims:
        for f in funcs:
            for algo in algos:
                for r in range(runs):
                    tasks.append((dim, f, algo, r))



    if getattr(CFG, "SHUFFLE_TASKS", False) and not CFG.TEST_MODE:
        rng = np.random.default_rng(CFG.BASE_SEED)
        rng.shuffle(tasks)
    return tasks


# Statistics
def summarize_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    ok = raw_df[raw_df["status"] == "ok"].copy()
    summary = (
        ok.groupby(["dimension", "function", "algorithm"])["best"]
        .agg(mean="mean", std="std", median="median", best="min", worst="max", n_runs="count")
        .reset_index()
    )
    return summary


def ranking_by_dimension(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim, df_dim in summary_df.groupby("dimension"):
        funcs = sorted(df_dim["function"].unique())
        algos = sorted(df_dim["algorithm"].unique())
        rank_records = []
        for f in funcs:
            df_f = df_dim[df_dim["function"] == f]
            means = []
            present_algos = []
            for algo in algos:
                vals = df_f.loc[df_f["algorithm"] == algo, "mean"].values
                if len(vals) == 1:
                    means.append(vals[0])
                    present_algos.append(algo)
            if len(means) >= 2:
                ranks = rankdata(means, method="average")
                for algo, rank in zip(present_algos, ranks):
                    rank_records.append({"dimension": dim, "function": f, "algorithm": algo, "rank": rank})
        rank_df = pd.DataFrame(rank_records)
        if not rank_df.empty:
            avg = rank_df.groupby(["dimension", "algorithm"])["rank"].mean().reset_index()
            rows.append(avg)
    if rows:
        return pd.concat(rows, ignore_index=True).sort_values(["dimension", "rank"])
    return pd.DataFrame(columns=["dimension", "algorithm", "rank"])


def win_tie_loss_against_clpso(summary_df: pd.DataFrame, target: str = "SCLPSO") -> pd.DataFrame:
    rows = []
    for dim, df_dim in summary_df.groupby("dimension"):
        algos = sorted(df_dim["algorithm"].unique())
        funcs = sorted(df_dim["function"].unique())
        for algo in algos:
            if algo == target:
                continue
            wins = ties = losses = 0
            ac_vals = []
            other_vals = []
            for f in funcs:
                ac = df_dim[(df_dim["function"] == f) & (df_dim["algorithm"] == target)]["mean"].values
                ot = df_dim[(df_dim["function"] == f) & (df_dim["algorithm"] == algo)]["mean"].values
                if len(ac) != 1 or len(ot) != 1:
                    continue
                ac_val = float(ac[0])
                ot_val = float(ot[0])
                ac_vals.append(ac_val)
                other_vals.append(ot_val)
                tol = 1e-12 * max(1.0, abs(ac_val), abs(ot_val))
                if ac_val + tol < ot_val:
                    wins += 1
                elif ot_val + tol < ac_val:
                    losses += 1
                else:
                    ties += 1
            p_value = np.nan
            if len(ac_vals) >= 5:
                try:

                    stat = wilcoxon(np.array(ac_vals), np.array(other_vals), zero_method="wilcox", alternative="two-sided")
                    p_value = float(stat.pvalue)
                except Exception:
                    p_value = np.nan
            rows.append({
                "dimension": dim,
                "competitor": algo,
                "SCLPSO_wins": wins,
                "ties": ties,
                "SCLPSO_losses": losses,
                "wilcoxon_p": p_value,
            })
    return pd.DataFrame(rows)


# Plots
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

def configure_plot_style() -> font_manager.FontProperties:


    font_path = CFG.FONT_PATH
    if not os.path.exists(font_path):
        raise FileNotFoundError(
            f"Required font file not found: {font_path}. "
            "Please put Times New Roman.ttf at this path or modify CFG.FONT_PATH."
        )

    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    prop.set_size(30)
    prop.set_weight("bold")
    font_name = prop.get_name()

    matplotlib.rcParams["font.family"] = font_name
    matplotlib.rcParams["font.sans-serif"] = [font_name]
    matplotlib.rcParams["font.serif"] = [font_name]
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = font_name
    matplotlib.rcParams["mathtext.it"] = f"{font_name}:italic"
    matplotlib.rcParams["mathtext.bf"] = f"{font_name}:bold"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["axes.formatter.use_mathtext"] = True
    matplotlib.rcParams["figure.dpi"] = CFG.PLOT_DPI


    matplotlib.rcParams["font.size"] = 30
    matplotlib.rcParams["axes.titlesize"] = 30
    matplotlib.rcParams["axes.labelsize"] = 30
    matplotlib.rcParams["xtick.labelsize"] = 30
    matplotlib.rcParams["ytick.labelsize"] = 30
    matplotlib.rcParams["legend.fontsize"] = 30
    matplotlib.rcParams["figure.titlesize"] = 30
    matplotlib.rcParams["font.weight"] = "bold"
    matplotlib.rcParams["axes.titleweight"] = "bold"
    matplotlib.rcParams["axes.labelweight"] = "bold"

    plt.rcParams.update(matplotlib.rcParams)
    return prop

def apply_paper_fonts(fig, prop: font_manager.FontProperties) -> None:

    for ax in fig.get_axes():
        text_items = [
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
            *ax.texts,
        ]
        for item in text_items:
            try:
                item.set_fontproperties(prop)
                item.set_fontsize(30)
                item.set_fontweight("bold")
            except Exception:
                pass

        legend = ax.get_legend()
        if legend is not None:
            for item in legend.get_texts():
                try:
                    item.set_fontproperties(prop)
                    item.set_fontsize(30)
                    item.set_fontweight("bold")
                except Exception:
                    pass
            title = legend.get_title()
            if title is not None:
                try:
                    title.set_fontproperties(prop)
                    title.set_fontsize(30)
                    title.set_fontweight("bold")
                except Exception:
                    pass

    for item in fig.texts:
        try:
            item.set_fontproperties(prop)
            item.set_fontsize(30)
            item.set_fontweight("bold")
        except Exception:
            pass


def _apply_axis_font(ax, prop: font_manager.FontProperties,
                     label_size: int = 30,
                     tick_size: int = 27,
                     title_size: int = 30) -> None:

    try:
        ax.title.set_fontproperties(prop)
        ax.title.set_fontsize(title_size)
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontproperties(prop)
        ax.xaxis.label.set_fontsize(label_size)
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontproperties(prop)
        ax.yaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontweight("bold")
    except Exception:
        pass

    ax.tick_params(axis="both", which="major", labelsize=tick_size, width=2, length=6)
    ax.tick_params(axis="both", which="minor", width=1, length=3)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        try:
            label.set_fontproperties(prop)
            label.set_fontsize(tick_size)
            label.set_fontweight("bold")
        except Exception:
            pass

def _safe_plot_name(final: bool, stem: str) -> str:
    return stem if final else stem.replace(".jpg", "_partial.jpg")


def _filter_plot_algorithms(df: pd.DataFrame) -> pd.DataFrame:


    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    allowed = set(CFG.ALGORITHMS)
    if "algorithm" in out.columns:
        out = out[out["algorithm"].isin(allowed)]
    if "competitor" in out.columns:
        out = out[out["competitor"].isin(allowed - {"SCLPSO"})]
    return out

def save_overall_ranking_table(rank_df: pd.DataFrame, out_root: str, final: bool = True) -> pd.DataFrame:
    rank_df = _filter_plot_algorithms(rank_df)
    if rank_df.empty:
        return pd.DataFrame(columns=["algorithm", "rank"])
    overall = (
        rank_df.groupby("algorithm", as_index=False)["rank"]
        .mean()
        .sort_values("rank", ascending=True)
        .reset_index(drop=True)
    )
    overall["overall_position"] = np.arange(1, len(overall) + 1)
    path = os.path.join(out_root, "overall_average_rankings.csv" if final else "overall_average_rankings_partial.csv")
    overall.to_csv(path, index=False)
    return overall


def plot_overall_average_rankings(rank_df: pd.DataFrame, out_root: str, final: bool = True) -> Optional[str]:
    rank_df = _filter_plot_algorithms(rank_df)
    overall = save_overall_ranking_table(rank_df, out_root, final=final)
    if overall.empty:
        return None

    prop = configure_plot_style()
    plot_data = overall.sort_values("rank", ascending=True)
    y_pos = np.arange(len(plot_data))

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#E41A1C" if algo == "SCLPSO" else "#377EB8" for algo in plot_data["algorithm"]]
    bars = ax.barh(y_pos, plot_data["rank"], color=colors, alpha=0.78)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data["algorithm"], fontproperties=prop, fontsize=30)
    ax.invert_yaxis()
    ax.set_xlabel("Average Rank", fontproperties=prop, fontsize=30, fontweight="bold")
    ax.set_ylabel("Algorithm", fontproperties=prop, fontsize=30, fontweight="bold")
    ax.set_title("Overall Average Ranking on CEC2017 Functions F1-F30",
                 fontproperties=prop, fontsize=30, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    for bar, rank_value in zip(bars, plot_data["rank"]):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{rank_value:.3f}", ha="left", va="center",
                fontproperties=prop, fontsize=30, fontweight="bold")

    apply_paper_fonts(fig, prop)
    plt.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_Overall_Average_Rankings.jpg"))
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

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, dim in enumerate(dims[:4]):
        ax = axes[idx]
        df_dim = rank_df[rank_df["dimension"] == dim].copy()
        df_dim = df_dim.set_index("algorithm").reindex(algos).dropna(subset=["rank"]).reset_index()
        y_pos = np.arange(len(df_dim))
        colors = ["#E41A1C" if algo == "SCLPSO" else "#377EB8" for algo in df_dim["algorithm"]]
        bars = ax.barh(y_pos, df_dim["rank"], color=colors, alpha=0.78)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_dim["algorithm"], fontproperties=prop, fontsize=30)
        ax.invert_yaxis()
        ax.set_xlabel("Average Rank", fontproperties=prop, fontsize=30, fontweight="bold")
        ax.set_ylabel("Algorithm", fontproperties=prop, fontsize=30, fontweight="bold")
        ax.set_title(f"{dim}D Average Ranking", fontproperties=prop, fontsize=30, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        for bar, rank_value in zip(bars, df_dim["rank"]):
            ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                    f"{rank_value:.2f}", ha="left", va="center",
                    fontproperties=prop, fontsize=30, fontweight="bold")

    for j in range(len(dims), len(axes)):
        axes[j].axis("off")

    apply_paper_fonts(fig, prop)
    plt.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_Dimension_Average_Rankings.jpg"))
    plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
    plt.close()
    return path


def _category_rankings(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = _filter_plot_algorithms(summary_df)
    if summary_df.empty:
        return pd.DataFrame(columns=["category", "algorithm", "rank"])

    groups = {
        "Unimodal": [1, 3],
        "Simple Multimodal": list(range(4, 11)),
        "Hybrid": list(range(11, 21)),
        "Composition": list(range(21, 31)),
    }

    rows = []
    for category, funcs in groups.items():
        df_cat = summary_df[summary_df["function"].isin(funcs)]
        if df_cat.empty:
            continue
        rank_records = []
        for (dim, func_num), df_func in df_cat.groupby(["dimension", "function"]):
            means = df_func[["algorithm", "mean"]].dropna().copy()
            if len(means) < 2:
                continue
            means["rank"] = rankdata(means["mean"].to_numpy(), method="average")
            means["dimension"] = dim
            means["function"] = func_num
            rank_records.append(means[["dimension", "function", "algorithm", "rank"]])
        if rank_records:
            cat_rank = pd.concat(rank_records, ignore_index=True)
            avg = cat_rank.groupby("algorithm", as_index=False)["rank"].mean()
            avg["category"] = category
            rows.append(avg[["category", "algorithm", "rank"]])
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["category", "algorithm", "rank"])


def plot_category_rankings(summary_df: pd.DataFrame, out_root: str, final: bool = True) -> Optional[str]:
    cat_df = _category_rankings(summary_df)
    if cat_df.empty:
        return None

    prop = configure_plot_style()
    categories = ["Unimodal", "Simple Multimodal", "Hybrid", "Composition"]
    algos = list(cat_df.groupby("algorithm")["rank"].mean().sort_values().index)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, category in enumerate(categories):
        ax = axes[idx]
        df_cat = cat_df[cat_df["category"] == category].set_index("algorithm").reindex(algos).dropna(subset=["rank"]).reset_index()
        y_pos = np.arange(len(df_cat))
        colors = ["#E41A1C" if algo == "SCLPSO" else "#377EB8" for algo in df_cat["algorithm"]]
        bars = ax.barh(y_pos, df_cat["rank"], color=colors, alpha=0.78)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_cat["algorithm"], fontproperties=prop, fontsize=30)
        ax.invert_yaxis()
        ax.set_xlabel("Average Rank", fontproperties=prop, fontsize=30, fontweight="bold")
        ax.set_ylabel("Algorithm", fontproperties=prop, fontsize=30, fontweight="bold")
        ax.set_title(f"{category} Functions", fontproperties=prop, fontsize=30, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        for bar, rank_value in zip(bars, df_cat["rank"]):
            ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                    f"{rank_value:.2f}", ha="left", va="center",
                    fontproperties=prop, fontsize=30, fontweight="bold")

    apply_paper_fonts(fig, prop)
    plt.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_Function_Category_Rankings.jpg"))
    plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
    plt.close()

    cat_path = os.path.join(out_root, "category_average_rankings.csv" if final else "category_average_rankings_partial.csv")
    cat_df.sort_values(["category", "rank"]).to_csv(cat_path, index=False)
    return path


def plot_win_tie_loss(wtl_df: pd.DataFrame, out_root: str, final: bool = True) -> Optional[str]:
    wtl_df = _filter_plot_algorithms(wtl_df)
    if wtl_df.empty:
        return None

    prop = configure_plot_style()
    agg = (
        wtl_df.groupby("competitor", as_index=False)[["SCLPSO_wins", "ties", "SCLPSO_losses"]]
        .sum()
    )
    agg["total"] = agg["SCLPSO_wins"] + agg["ties"] + agg["SCLPSO_losses"]
    agg = agg.sort_values(["SCLPSO_wins", "SCLPSO_losses"], ascending=[False, True]).reset_index(drop=True)

    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.bar(x, agg["SCLPSO_wins"], label="SCLPSO Wins", color="#4DAF4A", alpha=0.78)
    ax.bar(x, agg["ties"], bottom=agg["SCLPSO_wins"], label="Ties", color="#FF7F00", alpha=0.78)
    ax.bar(x, agg["SCLPSO_losses"], bottom=agg["SCLPSO_wins"] + agg["ties"],
           label="SCLPSO Losses", color="#377EB8", alpha=0.78)

    ax.set_xticks(x)
    ax.set_xticklabels(agg["competitor"], rotation=35, ha="right", fontproperties=prop, fontsize=30)
    ax.set_xlabel("Competitor", fontproperties=prop, fontsize=30, fontweight="bold")
    ax.set_ylabel("Number of Function-Dimension Cases", fontproperties=prop, fontsize=30, fontweight="bold")
    ax.set_title("Win/Tie/Loss Summary Against SCLPSO",
                 fontproperties=prop, fontsize=30, fontweight="bold")
    ax.legend(prop=prop, fontsize=30)
    ax.grid(True, alpha=0.3, axis="y")

    for i, row in agg.iterrows():
        total = int(row["total"])
        ax.text(i, total + 0.5, str(total), ha="center", va="bottom",
                fontproperties=prop, fontsize=30, fontweight="bold")

    apply_paper_fonts(fig, prop)
    plt.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_Win_Tie_Loss.jpg"))
    plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
    plt.close()
    return path


def _select_representative_curves(curve_rows: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    if not curve_rows:
        return []
    dims = sorted({r.get("dimension") for r in curve_rows if r.get("curve")})
    funcs = sorted({r.get("function") for r in curve_rows if r.get("curve")})
    target_dims = [30, 50, 100]
    target_funcs = [1, 10, 20, 30]
    selected = []
    for d in target_dims:
        if d not in dims:
            continue
        for f in target_funcs:
            if f in funcs:
                selected.append((d, f))
            if len(selected) >= 4:
                return selected

    for d in dims:
        for f in funcs:
            selected.append((d, f))
            if len(selected) >= 4:
                return selected
    return selected


def generate_algorithm_legend(out_root: str, final: bool = True) -> Optional[str]:

    if not final:
        return None
    prop = configure_plot_style()
    algos = list(CFG.ALGORITHMS)
    cmap = _algorithm_color_map(algos)
    fig, ax = plt.subplots(figsize=(8, 4))
    lines = []
    for algo in algos:
        line, = ax.plot([], [], label=algo, color=cmap.get(algo, "#377EB8"), linestyle="-", linewidth=2)
        lines.append(line)
    leg = ax.legend(handles=lines, loc="center", ncol=5, prop=prop, fontsize=27, framealpha=0.8)
    for txt in leg.get_texts():
        txt.set_fontproperties(prop)
        txt.set_fontsize(27)
        txt.set_fontweight("bold")
    ax.axis("off")
    fig.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_algorithm_legend.jpg"))
    fig.savefig(path, dpi=700, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_convergence_curves(curve_rows: List[Dict[str, Any]], out_root: str, final: bool = True) -> Optional[str]:

    if not final or not curve_rows:
        return None
    filtered = [r for r in curve_rows if r.get("curve")]
    if not filtered:
        return None
    selected_cases = _select_representative_curves(filtered)
    if not selected_cases:
        return None

    prop = configure_plot_style()
    algos = [a for a in CFG.ALGORITHMS if any(r.get("algorithm") == a for r in filtered)]
    colors = _algorithm_color_map(algos)

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    epsilon = 1e-30

    for ax_idx, (dim, func_num) in enumerate(selected_cases[:4]):
        ax = axes[ax_idx]
        plot_data_dict: Dict[str, np.ndarray] = {}
        all_valid_points: List[float] = []

        for algo in algos:
            x, y = _curve_error_arrays(filtered, int(dim), int(func_num), algo)
            if y is None or len(x) == 0:
                continue
            y = np.maximum(y, epsilon)
            plot_data_dict[algo] = (x, y)
            valid_mask = y > epsilon * 1.1
            if np.any(valid_mask):
                all_valid_points.extend(y[valid_mask].tolist())

        if not plot_data_dict:
            ax.axis("off")
            continue

        if all_valid_points:
            y_glob_min = max(min(all_valid_points), epsilon)
            if y_glob_min < 1e-20:
                y_glob_min = epsilon
            initial_values = [yy[0] for _, yy in plot_data_dict.values() if len(yy) > 0]
            y_glob_max = max(initial_values) if initial_values else max(all_valid_points)
            ax.set_ylim(y_glob_min * 0.1, y_glob_max * 2.0)

        for i, algo in enumerate(algos):
            if algo not in plot_data_dict:
                continue
            x, y = plot_data_dict[algo]
            ax.semilogy(x, y, color=colors.get(algo), linestyle="-", linewidth=2.5, label=algo)

        ax.yaxis.set_major_formatter(BoldMathFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.set_xlabel("Function Evaluations (FEs)", fontproperties=prop, fontsize=30, fontweight="bold")
        ax.set_ylabel("Mean Error (log scale)", fontproperties=prop, fontsize=30, fontweight="bold")
        ax.grid(True, which="major", alpha=0.4, linestyle="--", linewidth=1)
        ax.grid(True, which="minor", alpha=0.1, linestyle=":", linewidth=0.5)
        _apply_axis_font(ax, prop, label_size=30, tick_size=27, title_size=30)

    for j in range(len(selected_cases), len(axes)):
        axes[j].axis("off")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.08, wspace=0.28, hspace=0.34)
    path = os.path.join(out_root, "SCLPSO_Convergence_Curves.jpg")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path






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


def _algorithm_order_from_df(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty or "algorithm" not in df.columns:
        return list(CFG.ALGORITHMS)
    present = [a for a in CFG.ALGORITHMS if a in set(df["algorithm"].unique())]
    extras = sorted([a for a in df["algorithm"].unique() if a not in present])
    return present + extras


def _algorithm_color_map(algorithms: List[str]) -> Dict[str, str]:

    palette = [
        "#E41A1C",
        "#377EB8",
        "#FF8C00",
        "#984EA3",
        "#A65628",
        "#F781BF",
        "#666666",
        "#4DAF4A",
        "#984EA3",
        "#FFD700",
        "#1B9E77", "#D95F02", "#7570B3", "#66A61E", "#E6AB02",
    ]
    cmap: Dict[str, str] = {}
    for i, algo in enumerate(algorithms):
        cmap[algo] = palette[i % len(palette)]
    return cmap

def save_mean_std_reports(raw_df: pd.DataFrame, out_root: str, final: bool = True) -> List[str]:

    raw_df = _filter_plot_algorithms(raw_df)
    ok = raw_df[raw_df["status"] == "ok"].copy()
    if ok.empty:
        return []

    ok["error"] = ok.apply(lambda r: error_value(r["best"], int(r["function"])), axis=1)
    grouped = (
        ok.groupby(["dimension", "function", "algorithm"])
        .agg(
            raw_mean=("best", "mean"),
            raw_std=("best", "std"),
            error_mean=("error", "mean"),
            error_std=("error", "std"),
            median_error=("error", "median"),
            best_error=("error", "min"),
            worst_error=("error", "max"),
            n_runs=("error", "count"),
        )
        .reset_index()
        .sort_values(["dimension", "function", "algorithm"])
    )
    grouped["mean_std_error"] = grouped.apply(
        lambda r: f"{r['error_mean']:.2E} ± {r['error_std']:.2E}", axis=1
    )
    grouped["mean_std_raw"] = grouped.apply(
        lambda r: f"{r['raw_mean']:.2E} ± {r['raw_std']:.2E}", axis=1
    )

    suffix = "" if final else "_partial"
    long_path = os.path.join(out_root, f"mean_std_by_function_dimension{suffix}.csv")
    grouped.to_csv(long_path, index=False)

    wide_error = grouped.pivot_table(
        index=["dimension", "function"],
        columns="algorithm",
        values="mean_std_error",
        aggfunc="first",
    ).reset_index()
    algos = _algorithm_order_from_df(ok)
    columns = ["dimension", "function"] + [a for a in algos if a in wide_error.columns]
    wide_error = wide_error[columns]
    wide_path = os.path.join(out_root, f"mean_std_error_wide{suffix}.csv")
    wide_error.to_csv(wide_path, index=False)


    paper = wide_error.copy()
    paper["function"] = paper["function"].map(lambda x: f"F{int(x)}")
    paper_path = os.path.join(out_root, f"paper_mean_std_error_table{suffix}.csv")
    paper.to_csv(paper_path, index=False)

    return [long_path, wide_path, paper_path]


def wilcoxon_rank_sum_vs_clpso(raw_df: pd.DataFrame, out_root: str, final: bool = True) -> List[str]:

    raw_df = _filter_plot_algorithms(raw_df)
    ok = raw_df[raw_df["status"] == "ok"].copy()
    if ok.empty or "SCLPSO" not in set(ok["algorithm"]):
        return []

    ok["error"] = ok.apply(lambda r: error_value(r["best"], int(r["function"])), axis=1)
    rows = []
    for (dim, func_num), df_case in ok.groupby(["dimension", "function"]):
        ac = df_case[df_case["algorithm"] == "SCLPSO"]["error"].dropna().to_numpy(dtype=float)
        ac = ac[np.isfinite(ac)]
        if len(ac) < 2:
            continue
        for algo in sorted(df_case["algorithm"].unique()):
            if algo == "SCLPSO":
                continue
            other = df_case[df_case["algorithm"] == algo]["error"].dropna().to_numpy(dtype=float)
            other = other[np.isfinite(other)]
            if len(other) < 2:
                continue
            try:
                test = mannwhitneyu(ac, other, alternative="two-sided", method="auto")
                stat = float(test.statistic)
                p_val = float(test.pvalue)
            except Exception:
                stat = np.nan
                p_val = np.nan

            ac_mean = float(np.mean(ac))
            other_mean = float(np.mean(other))
            if np.isfinite(p_val) and p_val < 0.05:
                if ac_mean < other_mean:
                    conclusion = "SCLPSO significantly better"
                    sign = "+"
                elif ac_mean > other_mean:
                    conclusion = "SCLPSO significantly worse"
                    sign = "-"
                else:
                    conclusion = "No practical difference"
                    sign = "="
            else:
                conclusion = "No significant difference"
                sign = "="

            rows.append({
                "dimension": int(dim),
                "function": int(func_num),
                "competitor": algo,
                "test": "Wilcoxon rank-sum / Mann-Whitney U",
                "SCLPSO_mean_error": ac_mean,
                "competitor_mean_error": other_mean,
                "statistic": stat,
                "p_value": p_val,
                "significance_0.05": bool(np.isfinite(p_val) and p_val < 0.05),
                "sign": sign,
                "conclusion": conclusion,
                "n_SCLPSO": int(len(ac)),
                "n_competitor": int(len(other)),
            })

    if not rows:
        return []
    test_df = pd.DataFrame(rows).sort_values(["dimension", "function", "competitor"])
    suffix = "" if final else "_partial"
    test_path = os.path.join(out_root, f"wilcoxon_rank_sum_vs_SCLPSO{suffix}.csv")
    test_df.to_csv(test_path, index=False)

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
    summary_path = os.path.join(out_root, f"wilcoxon_rank_sum_summary_vs_SCLPSO{suffix}.csv")
    summary.to_csv(summary_path, index=False)
    return [test_path, summary_path]


def plot_wilcoxon_rank_sum_heatmap(raw_df: pd.DataFrame, out_root: str, final: bool = True) -> Optional[str]:

    paths = wilcoxon_rank_sum_vs_clpso(raw_df, out_root, final=final)
    if not paths:
        return None
    test_path = paths[0]
    try:
        test_df = pd.read_csv(test_path)
    except Exception:
        return None
    if test_df.empty:
        return None

    prop = configure_plot_style()

    score_map = {"+": 1.0, "=": 0.0, "-": -1.0}
    test_df["score"] = test_df["sign"].map(score_map).fillna(0.0)
    pivot = test_df.pivot_table(index="competitor", columns="dimension", values="score", aggfunc="mean")
    pivot = pivot.reindex([a for a in CFG.ALGORITHMS if a in pivot.index and a != "SCLPSO"])
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, max(6, 0.5 * len(pivot))))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c)}D" for c in pivot.columns], fontproperties=prop, fontsize=30)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontproperties=prop, fontsize=30)
    ax.set_xlabel("Dimension", fontproperties=prop, fontsize=30, fontweight="bold")
    ax.set_ylabel("Competitor", fontproperties=prop, fontsize=30, fontweight="bold")
    ax.set_title("Wilcoxon Rank-Sum Outcomes Against SCLPSO", fontproperties=prop, fontsize=30, fontweight="bold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if val > 0.25:
                label = "+"
            elif val < -0.25:
                label = "-"
            else:
                label = "="
            ax.text(j, i, label, ha="center", va="center", fontproperties=prop, fontsize=30, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean test outcome (+ better, = same, - worse)", fontproperties=prop, fontsize=30)
    apply_paper_fonts(fig, prop)
    plt.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_Wilcoxon_Rank_Sum_Heatmap.jpg"))
    plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
    plt.close()
    return path


def _curve_error_arrays(rows: List[Dict[str, Any]], dim: int, func_num: int, algo: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:

    selected = [
        r for r in rows
        if r.get("dimension") == dim
        and r.get("function") == func_num
        and r.get("algorithm") == algo
        and r.get("curve")
    ]
    if not selected:
        return np.array([]), None

    max_fes = int(CFG.MAX_FES_FACTOR * int(dim))



    common_fes = np.linspace(0, max_fes, 10000)
    x_plot = common_fes
    interpolated = []

    for r in selected:
        curve = r.get("curve", {})
        fes = np.asarray(curve.get("fes", []), dtype=np.float64)
        best = np.asarray(curve.get("best", []), dtype=np.float64)
        if len(fes) < 2 or len(best) < 2:
            continue
        order = np.argsort(fes)
        fes = fes[order]
        best = best[order]
        err = np.asarray([error_value(v, func_num) for v in best], dtype=np.float64)
        err = np.maximum(err, 0.0)
        finite = np.isfinite(err)
        if np.sum(finite) < 2:
            continue
        fill_value = np.nanmax(err[finite]) if np.any(finite) else 1.0
        err = np.nan_to_num(err, nan=fill_value, posinf=fill_value, neginf=0.0)
        interp = np.interp(common_fes, fes, err)
        window_size = max(3, min(21, int(len(interp) * 0.01)))
        interp = convolve_smooth_preserve_ends(interp, window_size)
        interpolated.append(interp)

    if not interpolated:
        return x_plot, None
    y = np.nanmedian(np.vstack(interpolated), axis=0)
    return x_plot, np.maximum(y, 1e-30)

def plot_all_function_convergence_curves(curve_rows: List[Dict[str, Any]], out_root: str, final: bool = True) -> List[str]:


    if not final or not curve_rows:
        return []
    filtered = [r for r in curve_rows if r.get("curve")]
    if not filtered:
        return []

    prop = configure_plot_style()
    algos = [a for a in CFG.ALGORITHMS if any(r.get("algorithm") == a for r in filtered)]
    colors = _algorithm_color_map(algos)
    dims = sorted({int(r["dimension"]) for r in filtered})
    funcs = sorted({int(r["function"]) for r in filtered})
    plot_dir = os.path.join(out_root, "convergence_curves_by_function")
    os.makedirs(plot_dir, exist_ok=True)
    paths: List[str] = []
    epsilon = 1e-30

    def plot_one_case(dim: int, func_num: int, plot_type: str = "normal") -> Optional[str]:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        all_valid_points: List[float] = []
        final_values: List[float] = []

        for algo in algos:
            x, y = _curve_error_arrays(filtered, int(dim), int(func_num), algo)
            if y is None or len(x) == 0:
                continue
            y = np.maximum(y, epsilon)
            plot_data_dict[algo] = (x, y)
            valid_mask = y > epsilon * 1.1
            if np.any(valid_mask):
                all_valid_points.extend(y[valid_mask].tolist())
            final_values.append(float(y[-1]))

        if not all_valid_points:
            plt.close(fig)
            return None

        y_glob_min = max(min(all_valid_points), epsilon)
        if y_glob_min < 1e-20:
            y_glob_min = epsilon
        initial_values = [yy[0] for _, yy in plot_data_dict.values() if len(yy) > 0]
        y_glob_max = max(initial_values) if initial_values else max(all_valid_points)
        ax.set_ylim(y_glob_min * 0.1, y_glob_max * 2.0)

        axins = None
        if plot_type == "zoomed":
            axins = inset_axes(
                ax, width="40%", height="30%", loc="center",
                bbox_to_anchor=(0.15, 0.15, 0.7, 0.7),
                bbox_transform=ax.transAxes,
            )

        max_len = 0
        for i, algo in enumerate(algos):
            if algo not in plot_data_dict:
                continue
            x, y = plot_data_dict[algo]
            max_len = max(max_len, len(x))
            color = colors.get(algo)
            ax.semilogy(x, y, color=color, linestyle="-", linewidth=2.5, label=algo)
            if axins is not None:
                axins.semilogy(x, y, color=color, linestyle="-", linewidth=2.5)

        if plot_type == "zoomed" and axins is not None and max_len > 0 and final_values:

            x_ref = next(iter(plot_data_dict.values()))[0]
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
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--", linewidth=0.8)

        ax.yaxis.set_major_formatter(BoldMathFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.set_xlabel("Function Evaluations (FEs)", fontproperties=prop, fontsize=30, fontweight="bold")
        ax.set_ylabel("Mean Error (log scale)", fontproperties=prop, fontsize=30, fontweight="bold")
        ax.grid(True, which="major", alpha=0.4, linestyle="--", linewidth=1)
        ax.grid(True, which="minor", alpha=0.1, linestyle=":", linewidth=0.5)
        _apply_axis_font(ax, prop, label_size=30, tick_size=27, title_size=30)

        suffix = "zoomed" if plot_type == "zoomed" else "log_error"
        path = os.path.join(plot_dir, f"CEC2017_{int(dim)}D_F{int(func_num):02d}_{suffix}.jpg")
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return path

    for dim in dims:
        for func_num in funcs:
            for plot_type in ("normal", "zoomed"):
                path = plot_one_case(int(dim), int(func_num), plot_type=plot_type)
                if path:
                    paths.append(path)
    return paths

def plot_all_function_boxplots(raw_df: pd.DataFrame, out_root: str, final: bool = True) -> List[str]:

    if not final:
        return []
    raw_df = _filter_plot_algorithms(raw_df)
    ok = raw_df[raw_df["status"] == "ok"].copy()
    if ok.empty:
        return []

    algos = [a for a in CFG.ALGORITHMS if a in set(ok["algorithm"].unique())]
    colors = _algorithm_color_map(algos)
    prop = configure_plot_style()
    plot_dir = os.path.join(out_root, "boxplots_by_function")
    os.makedirs(plot_dir, exist_ok=True)
    paths: List[str] = []

    for (dim, func_num), df_case in ok.groupby(["dimension", "function"]):
        fig, ax = plt.subplots(figsize=(12, 8))
        data: List[np.ndarray] = []
        labels: List[str] = []
        box_colors: List[str] = []

        for algo in algos:
            vals = df_case[df_case["algorithm"] == algo]["best"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) >= 2:
                data.append(vals)
                labels.append(algo)
                box_colors.append(colors.get(algo, "#377EB8"))

        if not data:
            plt.close(fig)
            continue

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
            meanprops={"color": "red", "linewidth": 1.5},
        )

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontproperties=prop, fontweight="bold", fontsize=30)

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for element in ["whiskers", "caps", "medians", "means"]:
            if element in bp:
                for item in bp[element]:
                    try:
                        item.set_linewidth(1.5)
                    except Exception:
                        pass

        if use_log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Objective Value (log scale)", fontproperties=prop, fontweight="bold", fontsize=30)
            ax.yaxis.set_major_formatter(BoldMathFormatter())
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        else:
            ax.set_ylabel("Objective Value", fontproperties=prop, fontweight="bold", fontsize=30)

        ax.grid(True, axis="y", alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=30, width=2, length=6)
        ax.tick_params(axis="both", which="minor", width=1, length=3)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(prop)
            label.set_fontsize(30)
            label.set_fontweight("bold")

        fig.tight_layout()
        path = os.path.join(plot_dir, f"CEC2017_{int(dim)}D_F{int(func_num):02d}_boxplot.jpg")
        fig.savefig(path, dpi=700, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths

def plot_average_rank_radar(rank_df: pd.DataFrame, out_root: str, final: bool = True) -> Optional[str]:

    rank_df = _filter_plot_algorithms(rank_df)
    if rank_df.empty:
        return None
    prop = configure_plot_style()
    dims = sorted(rank_df["dimension"].unique())
    algos = list(rank_df.groupby("algorithm")["rank"].mean().sort_values().index)
    if len(dims) < 2 or len(algos) < 2:
        return None

    pivot = rank_df.pivot_table(index="algorithm", columns="dimension", values="rank", aggfunc="mean").reindex(algos)
    max_rank = float(len(algos))

    score = (max_rank + 1.0 - pivot) / max_rank
    score = score.fillna(0.0)

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = _algorithm_color_map(algos)

    for algo in algos:
        values = score.loc[algo, dims].to_list()
        values += values[:1]
        lw = 2.6 if algo == "SCLPSO" else 1.3
        alpha = 0.95 if algo == "SCLPSO" else 0.55
        ax.plot(angles, values, linewidth=lw, label=algo, color=colors.get(algo), alpha=alpha)
        if algo == "SCLPSO":
            ax.fill(angles, values, color=colors.get(algo), alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{int(d)}D" for d in dims], fontproperties=prop, fontsize=30, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontproperties=prop, fontsize=30)
    ax.set_title("Comprehensive Radar Chart Based on Average Ranking Scores", fontproperties=prop, fontsize=30, fontweight="bold", pad=24)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, prop=prop, fontsize=30, frameon=False)
    apply_paper_fonts(fig, prop)
    plt.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_Average_Rank_Radar.jpg"))
    plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
    plt.close()
    return path


def compute_friedman_average_ranks(summary_df: pd.DataFrame,
                                   out_root: str,
                                   final: bool = True) -> pd.DataFrame:


    summary_df = _filter_plot_algorithms(summary_df)
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(columns=[
            "algorithm", "friedman_average_rank", "friedman_position",
            "friedman_chi_square", "friedman_p_value", "n_blocks"
        ])

    ok = summary_df.dropna(subset=["mean"]).copy()
    if ok.empty:
        return pd.DataFrame(columns=[
            "algorithm", "friedman_average_rank", "friedman_position",
            "friedman_chi_square", "friedman_p_value", "n_blocks"
        ])


    wide = ok.pivot_table(
        index=["dimension", "function"],
        columns="algorithm",
        values="mean",
        aggfunc="first",
    )

    algos = _algorithm_order_from_df(ok)
    algos = [a for a in algos if a in wide.columns]
    wide = wide[algos].dropna(axis=0, how="any")

    if wide.shape[0] == 0 or wide.shape[1] < 2:
        return pd.DataFrame(columns=[
            "algorithm", "friedman_average_rank", "friedman_position",
            "friedman_chi_square", "friedman_p_value", "n_blocks"
        ])

    rank_matrix = np.apply_along_axis(lambda row: rankdata(row, method="average"), 1, wide.to_numpy(dtype=float))
    avg_ranks = rank_matrix.mean(axis=0)

    chi_square = np.nan
    p_value = np.nan
    if wide.shape[0] >= 2 and wide.shape[1] >= 3:
        try:
            test = friedmanchisquare(*[wide[col].to_numpy(dtype=float) for col in wide.columns])
            chi_square = float(test.statistic)
            p_value = float(test.pvalue)
        except Exception:
            chi_square = np.nan
            p_value = np.nan

    friedman_df = pd.DataFrame({
        "algorithm": list(wide.columns),
        "friedman_average_rank": avg_ranks,
    }).sort_values("friedman_average_rank", ascending=True).reset_index(drop=True)
    friedman_df["friedman_position"] = np.arange(1, len(friedman_df) + 1)
    friedman_df["friedman_chi_square"] = chi_square
    friedman_df["friedman_p_value"] = p_value
    friedman_df["n_blocks"] = int(wide.shape[0])

    suffix = "" if final else "_partial"
    friedman_df.to_csv(os.path.join(out_root, f"friedman_average_ranks{suffix}.csv"), index=False)


    block_rank_df = pd.DataFrame(rank_matrix, columns=list(wide.columns), index=wide.index).reset_index()
    block_rank_df.to_csv(os.path.join(out_root, f"friedman_block_rank_matrix{suffix}.csv"), index=False)
    return friedman_df


def plot_friedman_average_rank_bar(summary_df: pd.DataFrame,
                                   out_root: str,
                                   final: bool = True) -> Optional[str]:


    friedman_df = compute_friedman_average_ranks(summary_df, out_root, final=final)
    if friedman_df.empty:
        return None

    prop = configure_plot_style()
    plot_data = friedman_df.sort_values("friedman_average_rank", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_data))
    colors = ["#E41A1C" if algo == "SCLPSO" else "#377EB8" for algo in plot_data["algorithm"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(y_pos, plot_data["friedman_average_rank"], color=colors, alpha=0.78)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data["algorithm"], fontproperties=prop, fontsize=30)
    ax.invert_yaxis()
    ax.set_xlabel("Friedman Average Rank", fontproperties=prop, fontsize=30, fontweight="bold")
    ax.set_ylabel("Algorithm", fontproperties=prop, fontsize=30, fontweight="bold")
    ax.set_title("Friedman Average Ranking on CEC2017 Functions F1-F30",
                 fontproperties=prop, fontsize=30, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    for bar, rank_value in zip(bars, plot_data["friedman_average_rank"]):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{rank_value:.3f}", ha="left", va="center",
                fontproperties=prop, fontsize=30, fontweight="bold")

    chi_square = plot_data["friedman_chi_square"].iloc[0]
    p_value = plot_data["friedman_p_value"].iloc[0]
    n_blocks = int(plot_data["n_blocks"].iloc[0])
    if np.isfinite(chi_square) and np.isfinite(p_value):
        stat_text = f"Friedman test: $\\chi^2$ = {chi_square:.3f}, p = {p_value:.3e}, blocks = {n_blocks}"
    else:
        stat_text = f"Friedman test: not available, blocks = {n_blocks}"
    ax.text(0.98, 0.02, stat_text, transform=ax.transAxes,
            ha="right", va="bottom", fontproperties=prop, fontsize=30,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="none"))

    apply_paper_fonts(fig, prop)
    plt.tight_layout()
    path = os.path.join(out_root, _safe_plot_name(final, "SCLPSO_Friedman_Average_Rank_Bar.jpg"))
    plt.savefig(path, dpi=CFG.PLOT_DPI, bbox_inches="tight")
    plt.close()
    return path

def generate_visual_outputs(raw_df: pd.DataFrame,
                            summary_df: pd.DataFrame,
                            rank_df: pd.DataFrame,
                            wtl_df: pd.DataFrame,
                            curve_rows: List[Dict[str, Any]],
                            out_root: str,
                            final: bool = True) -> List[str]:

    paths = []


    for table_func, args in [
        (save_mean_std_reports, (raw_df, out_root, final)),
        (wilcoxon_rank_sum_vs_clpso, (raw_df, out_root, final)),
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
        (plot_convergence_curves, (curve_rows, out_root, final)),
        (generate_algorithm_legend, (out_root, final)),
        (plot_average_rank_radar, (rank_df, out_root, final)),
        (plot_wilcoxon_rank_sum_heatmap, (raw_df, out_root, final)),
        (plot_all_function_convergence_curves, (curve_rows, out_root, final)),
        (plot_all_function_boxplots, (raw_df, out_root, final)),
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

def save_reports(raw_rows: List[Dict[str, Any]], curve_rows: List[Dict[str, Any]], out_root: str, final: bool = False):
    os.makedirs(out_root, exist_ok=True)
    raw_df = pd.DataFrame(raw_rows)
    raw_path = os.path.join(out_root, "raw_results.csv" if final else "raw_results_partial.csv")
    raw_df.to_csv(raw_path, index=False)

    if not raw_df.empty:
        summary_df = summarize_raw(raw_df)
        summary_path = os.path.join(out_root, "summary_results.csv" if final else "summary_results_partial.csv")
        summary_df.to_csv(summary_path, index=False)

        rank_df = ranking_by_dimension(summary_df)
        rank_path = os.path.join(out_root, "average_rankings.csv" if final else "average_rankings_partial.csv")
        rank_df.to_csv(rank_path, index=False)

        wtl_df = win_tie_loss_against_clpso(summary_df, target="SCLPSO")
        wtl_path = os.path.join(out_root, "win_tie_loss_vs_SCLPSO.csv" if final else "win_tie_loss_vs_SCLPSO_partial.csv")
        wtl_df.to_csv(wtl_path, index=False)

        if final:
            plot_paths = generate_visual_outputs(raw_df, summary_df, rank_df, wtl_df, curve_rows, out_root, final=True)
            if plot_paths:
                print(f"Generated visualization/table outputs: {len(plot_paths)} files", flush=True)
                for p in plot_paths[:20]:
                    print(f"  {p}", flush=True)
                if len(plot_paths) > 20:
                    print(f"  ... {len(plot_paths) - 20} additional per-function figures/tables saved under output subdirectories", flush=True)

    if final:
        curve_path = os.path.join(out_root, "curves.pkl")
        with open(curve_path, "wb") as f:
            pickle.dump(curve_rows, f)

    return raw_path


# Main
def main():
    os.makedirs(CFG.OUT_ROOT, exist_ok=True)


    with open(os.path.join(CFG.OUT_ROOT, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, ensure_ascii=False, indent=2)

    funcs = make_functions(CFG.EXCLUDE_F2)
    tasks = build_tasks()

    print("=" * 100)
    print("Revised SCLPSO / CEC2017 PSO-family experiment with F2 included and SCLPSO as the proposed algorithm")
    print(f"Shared library: {CFG.SO_PATH}")
    print(f"Output root:    {CFG.OUT_ROOT}")
    print(f"Dimensions:     {CFG.DIMENSIONS if not CFG.TEST_MODE else (10,)}")
    print(f"Functions:      {format_function_list(funcs) if not CFG.TEST_MODE else [1, 3]}")
    print(f"Exclude F2:     {CFG.EXCLUDE_F2}")
    if CFG.EXCLUDE_F2 and not CFG.TEST_MODE:
        print("F2 note:        F2 is excluded because it was removed from the official CEC2017 competition.")
    print(f"Runs:           {CFG.N_RUNS if not CFG.TEST_MODE else 2}")
    print(f"MaxFEs:         {CFG.MAX_FES_FACTOR} * D")
    print(f"Population:     {CFG.POP_SIZE}")
    print(f"Algorithms:     {CFG.ALGORITHMS if not CFG.TEST_MODE else ('SCLPSO', 'PSO')}")
    print(f"Tasks:          {len(tasks)}")
    print(f"CPU workers:    {CFG.MAX_PROCESSES} max; chunksize={CFG.CHUNKSIZE}; maxtasksperchild={CFG.TASKS_PER_CHILD}")
    print(f"Thread limits:  OMP={os.environ.get('OMP_NUM_THREADS')} MKL={os.environ.get('MKL_NUM_THREADS')} OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    print("=" * 100)

    if not os.path.exists(CFG.SO_PATH):
        raise FileNotFoundError(
            f"Cannot find {CFG.SO_PATH}. Please compile/copy the CEC2017 shared library first."
        )

    raw_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    if CFG.USE_PARALLEL and len(tasks) > 1:
        n_proc = min(CFG.MAX_PROCESSES, cpu_count() or 1, len(tasks))
        print(f"Running in parallel with {n_proc} processes.")
        with Pool(processes=n_proc, maxtasksperchild=CFG.TASKS_PER_CHILD) as pool:
            for i, (row, curve_row) in enumerate(
                pool.imap_unordered(run_one_task, tasks, chunksize=CFG.CHUNKSIZE), start=1
            ):
                raw_rows.append(row)
                curve_rows.append(curve_row)
                if i % CFG.SAVE_EVERY == 0:
                    path = save_reports(raw_rows, curve_rows, CFG.OUT_ROOT, final=False)
                    print(f"Progress {i}/{len(tasks)} | partial saved: {path}", flush=True)
    else:
        print("Running sequentially.")
        for i, task in enumerate(tasks, start=1):
            row, curve_row = run_one_task(task)
            raw_rows.append(row)
            curve_rows.append(curve_row)
            if i % CFG.SAVE_EVERY == 0:
                path = save_reports(raw_rows, curve_rows, CFG.OUT_ROOT, final=False)
                print(f"Progress {i}/{len(tasks)} | partial saved: {path}", flush=True)

    final_raw_path = save_reports(raw_rows, curve_rows, CFG.OUT_ROOT, final=True)

    print("=" * 100)
    print("Experiment finished.")
    print(f"Raw results:             {final_raw_path}")
    print(f"Summary results:         {os.path.join(CFG.OUT_ROOT, 'summary_results.csv')}")
    print(f"Average rankings:        {os.path.join(CFG.OUT_ROOT, 'average_rankings.csv')}")
    print(f"Win/tie/loss vs SCLPSO:   {os.path.join(CFG.OUT_ROOT, 'win_tie_loss_vs_SCLPSO.csv')}")
    print(f"Mean±STD error table:    {os.path.join(CFG.OUT_ROOT, 'mean_std_error_wide.csv')}")
    print(f"Wilcoxon rank-sum test:  {os.path.join(CFG.OUT_ROOT, 'wilcoxon_rank_sum_vs_SCLPSO.csv')}")
    print(f"Per-function curves dir: {os.path.join(CFG.OUT_ROOT, 'convergence_curves_by_function')}")
    print(f"Per-function boxplot dir:{os.path.join(CFG.OUT_ROOT, 'boxplots_by_function')}")
    print(f"Curves:                  {os.path.join(CFG.OUT_ROOT, 'curves.pkl')}")
    print(f"Overall ranking figure:  {os.path.join(CFG.OUT_ROOT, 'SCLPSO_Overall_Average_Rankings.jpg')}")
    print(f"Friedman ranking fig.: {os.path.join(CFG.OUT_ROOT, 'SCLPSO_Friedman_Average_Rank_Bar.jpg')}")
    print(f"Dimension ranking fig.:  {os.path.join(CFG.OUT_ROOT, 'SCLPSO_Dimension_Average_Rankings.jpg')}")
    print(f"Category ranking fig.:   {os.path.join(CFG.OUT_ROOT, 'SCLPSO_Function_Category_Rankings.jpg')}")
    print(f"Win/tie/loss figure:     {os.path.join(CFG.OUT_ROOT, 'SCLPSO_Win_Tie_Loss.jpg')}")
    print(f"Convergence figure:      {os.path.join(CFG.OUT_ROOT, 'SCLPSO_Convergence_Curves.jpg')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
