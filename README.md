# Stackelberg–Potential Game Framework — Reference Implementation

**Package B — code only.** Self-contained implementation of the game layer with
its self-tests, without the recorded data products. Requires Python 3.8+ and
NumPy.

Companion code for the IEEE Transactions on Cybernetics manuscript

> **Stackelberg–Potential Framework for Dispatch-Free Multi-UAV Coordination**
> Tian Chen, Rui Zhang, Yanxian Bi, Kemi Xu, Xiangwang Hou, Jingjing Wang,
> C. L. Philip Chen

The package contains a self-contained implementation of the game layer.
Section, equation, theorem and table numbers refer to the manuscript as
submitted.

---

## Contents

| File | Description |
|---|---|
| `sop_core.py` | The game layer: HOUBO cost model, exact potential `Φ_ex` of Eq. (14), reporting potential `Φ` of Eq. (16), best-response dynamics, Stackelberg price loop, projection onto `P`. NumPy only, 1 505 lines. |
| `test_sop_core.py` | 13 self-tests for the invariants the analysis relies on. |
| `requirements.txt` | Runtime dependency (NumPy). |
| `LICENSE` | MIT licence. |

---

## Quick start

```bash
pip install -r requirements.txt

python sop_core.py                      # equilibrium trace of the primary instance
python sop_core.py --lemma              # Lemma 1 on its own verification instance
python test_sop_core.py                 # 13 self-tests
```

The three commands take about a minute in total. Expected output of the first one:

```
[Sec. III-A] grid cells          : 400
[Sec. III-A] UAVs / hubs         : 3 / 2
[Sec. IV-B] rollout horizon N_T : 120 steps
[Sec. IV-D] hubs                : Hub0(c=0.50, rho=8.0, lam=1.0), Hub1(c=0.30, rho=10.0, lam=2.0)
[Sec. IV-C] Phi    at Phase-0 init : 7.9835e+04   (reporting, Eq. (16))
[Sec. IV-C] Phi_ex at Phase-0 init : 7.1340e+04   (Lemma 1,   Eq. (14))

[Sec. IV-F] Stackelberg outer rounds : 4
[Sec. V-B] BRD iterations (mean)  : 2.5
[Sec. IV-C] Phi trajectory (Eq.16): 2.222e+04 -> 2.237e+04 -> 2.163e+04 -> 2.163e+04
[Sec. IV-C] Phi_ex end / start    : 1.2491e+04 / 7.1340e+04  (fall 5.71x)
[Lemma 1]   verification         : run `python sop_core.py --lemma` for 90/90 exact and 84/90 sign
[Sec. V-B]  non-increasing Phi sweeps: 3/4   (the Eq. (16) charge can mask a move; Phi_ex cannot)
[Theorem 2] Hub0 final price     : c=0.57, rho=8.07, lam=1.00
[Theorem 2] Hub1 final price     : c=0.84, rho=10.04, lam=2.00
[Sec. V-C]  Theorem 2 hyp.        : P box True | U_i affine in p True (1.8e-12)
[Sec. V-C]  Theorem 2 hyp.        : U_hub affine in p True (1.4e-14) | NE at final price True
```

The run then prints an equilibrium summary: claimed and contested cell counts,
the two potentials, and the per-UAV assignment of role and hub.

The trace prints two potentials, and Sec. V uses them for different things.
`Phi` is the reporting potential of Eq. (16), which sums the per-cell charge
`Ω`; it may rise between outer rounds (2.222e+04 → 2.237e+04) because the leader
moves first and the fleet re-equilibrates. `Phi_ex` is the exact potential of
Eq. (14), the object of Lemma 1; it falls on every accepted improving move and
by 5.71× over the run.

---

## Code-to-paper map

| Manuscript | Object | Code |
|---|---|---|
| Sec. III-A, Eq. (1) | decision grid `G` | `build_grid`, `GridPoint` |
| Sec. IV-A, Eq. (9) | HOUBO polynomial `f(x) = Σ c_p Π x_n` | `HOUBO` |
| Sec. IV-A | variable mapping `x_ijk(t)`, pruning | `SeedingProblem` |
| Sec. IV-A | `J_acc`, `J_time` | `HOUBOCosts.f_acc`, `.f_time` |
| Sec. IV-A, Eq. (9d) | conflict charge `t(c) = γ(c-1)²/c` | `_conflict_share` |
| Sec. IV-B | trajectory and propulsion model | `TrajectoryModel` |
| Sec. IV-C, Eq. (12) | utility `U_i`, delivery bonus `b_i` | `OrdinalPotentialGame.utility` |
| Sec. IV-C, Eq. (13) | transfer payment `C_hub` | `_hub_access_cost` |
| Sec. IV-C, Eq. (14) | exact potential `Φ_ex` | `OrdinalPotentialGame.exact_potential`, `_phi_ex` |
| Sec. IV-C, Eqs. (15)–(16) | `Ω̃` and `Ω` | `conflict_aggregates` |
| Sec. IV-C, Eq. (16) | reporting potential `Φ` | `OrdinalPotentialGame.potential` |
| Sec. IV-D, Eq. (17) | price box `P`, projection `Π_P` | `PRICE_LOWER`, `PRICE_UPPER`, `_project_price` |
| Sec. IV-D, Eq. (18) | hub utility `U_hub,m` | `StackelbergGame.hub_utility` |
| Sec. IV-D, Eq. (19) | leader's program `V(p)` | `StackelbergGame.hub_gradient`, `.solve` |
| Sec. IV-F, Algorithm 1 | SoP-Seeding, lines 1–13 | `StackelbergGame.solve` |
| Lemma 1 | `ΔU_i = -ΔΦ_ex` | `verify_lemma1_exact`, `sample_paper_lemma1_exact` |
| Theorem 1 | BRD reaches a pure NE | `BestResponseDynamics.iterate` |
| Theorem 2 | existence, via Berge's maximum theorem | `verify_theorem2_hypotheses` |
| Corollary 1 | `p^(k) ∈ P`, termination within the iteration budget | `_project_price`, the `epsilon_SE` loop |

The `Sec. X-Y` anchors also appear as comments in `sop_core.py`, so a reader can
jump from a proof to the lines that implement it.

---

## Lemma 1 and the scope of the sign identity

`python sop_core.py --lemma` runs the verification instance of the Lemma and
evaluates 90 unilateral deviations, 30 per player, drawn from the four families
of Algorithm 1. With the marginals `Ω̃` of Eq. (15) in place of the charge `Ω` of
Eq. (16), the potential moves exactly against the deviating player's utility:

```
exact identity  dU + dPhi_ex = 0       : 90/90 = 100.0%
    add a cell    : 21/21  worst residual 1.5e-16
    drop a cell   : 29/29  worst residual 1.3e-14
    flip the role : 23/23  worst residual 1.8e-15
    switch hub    : 17/17  worst residual 4.3e-15
```

The weaker sign identity holds for the reporting potential of Eq. (16):

```
sign identity  sign(dU) = -sign(dPhi) : 84/90 = 93.3%
    add a cell    : 18/18
    drop a cell   : 26/32
    flip the role : 18/18
    switch hub    : 22/22
```

Every exception is a drop-a-cell move. Dropping a cell changes the private
reward `R_task·n_i`, which does not enter `Φ`; the leftover term
`T = Δ(R_task n_i) - Δ(C_hub,i) + Σ_{k≠i} Δ(f_pen,k)` then decides the sign, and
with `R_task = 0` the same instance yields 82/82. This is the reason the
manuscript states Lemma 1 for `Φ_ex` and reports `Φ` separately, and the probe
prints both rates family by family rather than as a single number.

---

## Best responses and the leader's price

**Theorem 1.** `BestResponseDynamics.iterate` terminates at a profile with no
improving one-bit deviation, and `Φ_ex` is non-increasing within every pricing
round. Both properties are asserted as tests.

**Corollary 1.** The projection is the code path `solve` takes, and a test
asserts that every price iterate stays inside
`P = [0.1, 200] × [0.1, 100] × [0.1, 100]`.

**Theorem 2.** The proof rests on Berge's maximum theorem rather than on a
fixed-point argument, so the diagnostic measures what a single run can observe:
the price box and its iterates, the affinity of `U_i` and `U_hub` in every price
coordinate (second differences of 1.8e-12 and 1.4e-14, which is stronger than the
continuity the proof assumes), and the absence of an improving deviation at the
reported price. The upper semicontinuity of `V(p)` is analytic and carries no
numerical diagnostic.

---

## Terminal accuracy and the SQP refinement

The accuracy columns of the manuscript tables are produced by the sequential
least-squares SQP refinement of Sec. IV-B, which re-solves the terminal
trajectories of the allocation returned by the game layer. That refinement is a
component in its own right and is not part of this package. What the package
carries is the game layer, driven by the geometric trajectory controller of
Sec. IV-B.

| Table | Column produced by the SQP refinement |
|---|---|
| Table I | `SQP feasible (%)`, `SQP error (mm)`; the `discrete error (m)` column is the geometric rollout |
| Tables II–IV | terminal energy, response time and accuracy of each row |
| Table SI | `accuracy (mm)` |
| Table SII | `accuracy (m)` |
| Table SIII | `accuracy (mm)` |
| Tables V–VI | none; these report wall-clock time and served counts only |

Terminal accuracy in a run of `sop_core.py` is the accuracy of the geometric
rollout, that is, the quantity behind the `discrete error (m)` column of Table I.
Reading an SQP-refined figure next to a run of this code compares two solvers;
the table above marks where that applies. Every other quantity of Sec. VI —
coverage, assignment conflicts, urgent share, prices, potentials, iteration
counts and all wall-clock times — is produced by the code here.

Theorem 3 assumes the quasi-Newton matrices of that solver to be uniformly
positive definite, so hypothesis (iii) enters the analysis as a standing
assumption, exactly as the manuscript states it.

---

## Reproducibility

Every stochastic step is seeded:

* the grid layout, including the spatial mixture of the two seed varieties, is
  drawn from a private RNG keyed by `grid_seed`;
* the initial UAV positions, initial claims, best-response ordering and candidate
  sampling draw from the global `random` module, which `build_scenario` seeds from
  `seed`;
* the Lemma 1 instance in `PAPER_LEMMA_PARAMS` is re-seeded on entry, so
  `python sop_core.py --lemma` prints the same counts on every run;
* `python sop_core.py` is byte-identical across runs: 1 936 bytes, md5
  `4bc904cd5ed0`. `test_same_seed_reproduces_equilibrium` asserts it for the
  equilibrium itself.

Single-threaded BLAS gives exact agreement with the numbers quoted above:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 python sop_core.py
```

The released core was cross-checked against the authors' full research
implementation on an identical scenario and seed: potential at Phase-0
initialisation, potential history across Stackelberg rounds, outer-round count,
mean best-response iteration count, hub price history and the final allocation
agree exactly, and the per-deviation utilities agree to 1e-9.

---

## Model conventions

**Trajectory layer.** `sop_core.py` drives the game layer with the geometric
trajectory controller of Sec. IV-B, which rolls out the translational dynamics
and integrates the propulsion power model.

**Price step.** Algorithm 1 lines 10–11 use a deterministic diminishing step
`η_k = η_0 / (1 + βk)` applied to a forward difference of the leader's own
utility, taken after a full lower-level sweep. The step is not stochastic, so no
stochastic-approximation argument is invoked.

**Physical dispersion.** `w = 0` in all experiments: the travel-energy and
flight-time terms already penalise distance from a hub, so no explicit dispersion
term is added. The potential is stated so that any additive private term
`w·‖x̄_i − h_τi‖` preserves Lemma 1 for `w ≥ 0`.

---

## Citation

If you use this code or these products, please cite the manuscript above. A
BibTeX entry will be added once the paper appears in print.

## License

Released under the MIT Licence. See `LICENSE`.
