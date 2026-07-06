# -*- coding: utf-8 -*-
"""
Standalone SCLPSO implementation.

The optimizer uses:
- Halton-chaotic initialization
- dimension-wise comprehensive learning
- success-history-guided perturbation
- linear population size reduction
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.stats import qmc


Array = np.ndarray
Objective = Callable[[Array], float]


@dataclass
class SCLPSOConfig:
    pop_size: int = 30
    n_init_factor: float = 6.0
    max_pop: int = 720
    n_min: int = 30

    cl_refresh_gap: int = 7
    cl_c: float = 1.49445
    gbest_c_min: float = 0.05
    gbest_c_max: float = 0.85

    hist_size: int = 6
    p_best_rate: float = 0.11
    archive_rate: float = 2.6
    perturb_rate: float = 0.80

    curve_points: int = 200
    lower_bound: float = -100.0
    upper_bound: float = 100.0


class BudgetedObjective:
    def __init__(self, func: Objective, max_fes: int):
        self.func = func
        self.max_fes = int(max_fes)
        self.fes = 0

    def remaining(self) -> int:
        return max(0, self.max_fes - self.fes)

    def eval_pop(self, pop: Array) -> Array:
        pop = np.asarray(pop, dtype=np.float64)
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)

        out = np.full(pop.shape[0], np.inf, dtype=np.float64)
        n_eval = min(pop.shape[0], self.remaining())
        for i in range(n_eval):
            out[i] = float(self.func(pop[i]))
        self.fes += n_eval
        return out


def reflect_bounds(x: Array, bounds: Tuple[float, float]) -> Array:
    lb, ub = bounds
    y = np.asarray(x, dtype=np.float64).copy()
    y = np.where(y > ub, 2.0 * ub - y, y)
    y = np.where(y < lb, 2.0 * lb - y, y)
    return np.clip(y, lb, ub)


def clamp_velocity(v: Array, bounds: Tuple[float, float], ratio: float = 0.2) -> Array:
    lb, ub = bounds
    vmax = ratio * (ub - lb)
    return np.clip(v, -vmax, vmax)


def halton_chaos_init(n: int, dim: int, bounds: Tuple[float, float], rng: np.random.Generator) -> Array:
    lb, ub = bounds

    sampler = qmc.Halton(d=dim, scramble=True, seed=int(rng.integers(1, 2**31 - 1)))
    halton = qmc.scale(sampler.random(n=n), lb, ub)

    chaos = rng.random((n, dim))
    for _ in range(5):
        chaos = 4.0 * chaos * (1.0 - chaos)
    chaos = lb + chaos * (ub - lb)

    return np.clip(0.7 * halton + 0.3 * chaos, lb, ub).astype(np.float64)


def learning_probability(n: int) -> Array:
    if n <= 1:
        return np.array([0.05], dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    return 0.05 + 0.45 * (np.exp(10.0 * idx / (n - 1.0)) - 1.0) / (np.exp(10.0) - 1.0)


def sample_f(memory_f: float, rng: np.random.Generator) -> float:
    for _ in range(50):
        value = memory_f + 0.1 * math.tan(math.pi * (rng.random() - 0.5))
        if value > 0.0:
            return float(min(value, 1.0))
    return 0.5


def weighted_lehmer_mean(values: Array, weights: Array) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weights * values))
    if denom <= 1e-300:
        return float(np.mean(values))
    return float(np.sum(weights * values * values) / denom)


def shade_boundary_correction(mutant: Array, parent: Array, bounds: Tuple[float, float]) -> Array:
    lb, ub = bounds
    y = np.asarray(mutant, dtype=np.float64).copy()
    lower = y < lb
    upper = y > ub
    y[lower] = 0.5 * (parent[lower] + lb)
    y[upper] = 0.5 * (parent[upper] + ub)
    return np.clip(y, lb, ub)


def record_curve(curve: Dict[str, List[float]], fes: int, best: float,
                 max_fes: int, next_fes: int, interval: int) -> int:
    if fes >= next_fes or fes >= max_fes:
        curve["fes"].append(int(fes))
        curve["best"].append(float(best))
        return next_fes + interval
    return next_fes


def sclpso(
    objective: Objective,
    dim: int,
    max_fes: int,
    bounds: Tuple[float, float] = (-100.0, 100.0),
    seed: int = 20260521,
    config: Optional[SCLPSOConfig] = None,
) -> Dict[str, Any]:
    if config is None:
        config = SCLPSOConfig()

    rng = np.random.default_rng(seed)
    evaluator = BudgetedObjective(objective, max_fes)

    n_min = max(4, int(config.n_min))
    n_min = max(n_min, min(config.pop_size, int(config.max_pop)))

    n_init = int(round(config.n_init_factor * dim))
    n_init = max(config.pop_size, n_init, n_min)
    n_init = min(int(config.max_pop), n_init)
    n_init = min(n_init, max(n_min, max_fes // 10))

    particles = halton_chaos_init(n_init, dim, bounds, rng)
    velocities = np.zeros_like(particles)

    fitness = evaluator.eval_pop(particles)
    pbest_pos = particles.copy()
    pbest_val = fitness.copy()

    best_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[best_idx].copy()
    best_fit = float(pbest_val[best_idx])

    n = particles.shape[0]
    pc = learning_probability(n)
    no_improve = np.zeros(n, dtype=np.int64)
    exemplars = np.tile(np.arange(n)[:, None], (1, dim))

    def refresh_particle(i: int) -> None:
        nonlocal exemplars, pc, pbest_val
        n_local = pbest_pos.shape[0]
        if n_local <= 1:
            exemplars[i, :] = i
            return

        all_self = True
        for d in range(dim):
            if rng.random() < pc[i] and n_local >= 3:
                a = int(rng.integers(0, n_local))
                while a == i:
                    a = int(rng.integers(0, n_local))
                b = int(rng.integers(0, n_local))
                while b == i or b == a:
                    b = int(rng.integers(0, n_local))
                exemplars[i, d] = a if pbest_val[a] < pbest_val[b] else b
                all_self = False
            else:
                exemplars[i, d] = i

        if all_self and n_local >= 3:
            d = int(rng.integers(0, dim))
            a = int(rng.integers(0, n_local))
            while a == i:
                a = int(rng.integers(0, n_local))
            b = int(rng.integers(0, n_local))
            while b == i or b == a:
                b = int(rng.integers(0, n_local))
            exemplars[i, d] = a if pbest_val[a] < pbest_val[b] else b

    def refresh_all() -> None:
        for i in range(pbest_pos.shape[0]):
            refresh_particle(i)

    refresh_all()

    h_size = int(config.hist_size)
    memory_cr = np.full(h_size, 0.5, dtype=np.float64)
    memory_f = np.full(h_size, 0.5, dtype=np.float64)
    memory_pos = 0
    archive = np.empty((0, dim), dtype=np.float64)

    interval = max(1, max_fes // max(10, int(config.curve_points)))
    next_record = 0
    curve = {"fes": [], "best": []}
    next_record = record_curve(curve, evaluator.fes, best_fit, max_fes, next_record, interval)

    while evaluator.fes < max_fes:
        n = particles.shape[0]
        progress = evaluator.fes / max_fes

        for i in range(n):
            if no_improve[i] >= config.cl_refresh_gap:
                refresh_particle(i)
                no_improve[i] = 0

        dims = np.arange(dim)
        exemplar_pos = np.empty_like(particles)
        for i in range(n):
            exemplar_pos[i] = pbest_pos[exemplars[i], dims]

        w = 0.9 - 0.5 * progress
        c_g = config.gbest_c_min + (config.gbest_c_max - config.gbest_c_min) * progress

        velocities = (
            w * velocities
            + config.cl_c * rng.random((n, dim)) * (exemplar_pos - particles)
            + c_g * rng.random((n, dim)) * (gbest_pos[None, :] - particles)
        )
        velocities = clamp_velocity(velocities, bounds, ratio=0.2)
        particles = reflect_bounds(particles + velocities, bounds)

        new_fit = evaluator.eval_pop(particles)
        finite = np.isfinite(new_fit)
        fitness[finite] = new_fit[finite]

        improved = new_fit < pbest_val
        pbest_pos[improved] = particles[improved]
        pbest_val[improved] = new_fit[improved]
        no_improve[improved] = 0
        no_improve[~improved] += 1

        best_idx = int(np.argmin(pbest_val))
        if pbest_val[best_idx] < best_fit:
            best_fit = float(pbest_val[best_idx])
            gbest_pos = pbest_pos[best_idx].copy()

        if evaluator.remaining() > 0 and n >= 4:
            n_trials = max(1, int(round(config.perturb_rate * n)))
            n_trials = min(n_trials, evaluator.remaining(), n)
            trial_indices = rng.choice(n, size=n_trials, replace=False)

            order = np.argsort(pbest_val)
            p_num = max(2, min(n, int(round(config.p_best_rate * n))))
            top_indices = order[:p_num]

            union = particles if archive.shape[0] == 0 else np.vstack([particles, archive])
            union_size = union.shape[0]

            trials = np.empty((n_trials, dim), dtype=np.float64)
            cr_values = np.empty(n_trials, dtype=np.float64)
            f_values = np.empty(n_trials, dtype=np.float64)

            for k, i in enumerate(trial_indices):
                mem_idx = int(rng.integers(0, h_size))
                cr = 0.0 if np.isnan(memory_cr[mem_idx]) else float(np.clip(rng.normal(memory_cr[mem_idx], 0.1), 0.0, 1.0))
                f = sample_f(float(memory_f[mem_idx]), rng)

                cr_values[k] = cr
                f_values[k] = f

                pbest_idx = int(rng.choice(top_indices))
                r1 = int(rng.integers(0, n))
                while r1 == i:
                    r1 = int(rng.integers(0, n))

                r2 = int(rng.integers(0, union_size))
                while r2 == i or r2 == r1:
                    r2 = int(rng.integers(0, union_size))

                mutant = (
                    particles[i]
                    + f * (pbest_pos[pbest_idx] - particles[i])
                    + f * (particles[r1] - union[r2])
                )
                mutant = shade_boundary_correction(mutant, particles[i], bounds)

                cross = rng.random(dim) <= cr
                cross[int(rng.integers(0, dim))] = True
                trials[k] = np.where(cross, mutant, particles[i])

            trial_fit = evaluator.eval_pop(trials)

            success_cr: List[float] = []
            success_f: List[float] = []
            delta_f: List[float] = []
            new_archive: List[Array] = []

            for k, i in enumerate(trial_indices):
                if not np.isfinite(trial_fit[k]):
                    continue

                if trial_fit[k] <= fitness[i]:
                    if trial_fit[k] < fitness[i]:
                        new_archive.append(particles[i].copy())
                        success_cr.append(float(cr_values[k]))
                        success_f.append(float(f_values[k]))
                        delta_f.append(float(abs(fitness[i] - trial_fit[k])))

                    particles[i] = trials[k]
                    fitness[i] = trial_fit[k]

                if trial_fit[k] < pbest_val[i]:
                    pbest_pos[i] = trials[k]
                    pbest_val[i] = trial_fit[k]
                    no_improve[i] = 0
                    if trial_fit[k] < best_fit:
                        best_fit = float(trial_fit[k])
                        gbest_pos = trials[k].copy()

            if new_archive:
                archive = np.vstack([archive, np.asarray(new_archive, dtype=np.float64)])

            archive_max = max(0, int(round(config.archive_rate * n)))
            if archive.shape[0] > archive_max:
                keep = rng.choice(archive.shape[0], size=archive_max, replace=False)
                archive = archive[keep]

            if success_f:
                cr_arr = np.asarray(success_cr, dtype=np.float64)
                f_arr = np.asarray(success_f, dtype=np.float64)
                df_arr = np.asarray(delta_f, dtype=np.float64)
                weights = df_arr / np.sum(df_arr) if np.sum(df_arr) > 0 else np.full(len(df_arr), 1.0 / len(df_arr))

                memory_cr[memory_pos] = np.nan if np.max(cr_arr) <= 0.0 else weighted_lehmer_mean(cr_arr, weights)
                memory_f[memory_pos] = weighted_lehmer_mean(f_arr, weights)
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

            pc = learning_probability(target_n)
            exemplars = np.tile(np.arange(target_n)[:, None], (1, dim))
            refresh_all()

            archive_max = max(0, int(round(config.archive_rate * target_n)))
            if archive.shape[0] > archive_max:
                keep_a = rng.choice(archive.shape[0], size=archive_max, replace=False)
                archive = archive[keep_a]

        next_record = record_curve(curve, evaluator.fes, best_fit, max_fes, next_record, interval)

    return {
        "best_x": gbest_pos,
        "best_y": float(best_fit),
        "fes": int(evaluator.fes),
        "curve": curve,
    }


if __name__ == "__main__":
    def sphere(x: Array) -> float:
        return float(np.sum(x * x))

    result = sclpso(
        objective=sphere,
        dim=30,
        max_fes=300000,
        bounds=(-100.0, 100.0),
        seed=20260521,
    )

    print("Best fitness:", result["best_y"])
    print("FEs:", result["fes"])
