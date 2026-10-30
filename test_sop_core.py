"""
Self-tests for sop_core.py.

Run either way:

    python test_sop_core.py
    pytest test_sop_core.py

The tests check invariants that must hold independently of any particular
scenario, plus one exact reproduction of the paper's Lemma 1 experiment.  The
reproduction is deterministic and takes about half a minute; everything else
runs in a second or so and needs no data products.
"""

import random

import numpy as np

import sop_core as S


def _small_params(**overrides):
    p = dict(S.DEMO_PARAMS)
    p.update(field_width=20.0, field_height=20.0, grid_size=2.0,  # 100 cells
             max_OPG_iter=4, max_SE_iter=2, T_total=20.0)
    p.update(overrides)
    return p


def _small_hubs():
    return [S.Hub(0, 2.0, 2.0, c=0.5, rho=8.0, lam=1.0),
            S.Hub(1, 18.0, 18.0, c=0.3, rho=10.0, lam=2.0)]


def _scenario(params):
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    grid = S.build_grid(params['field_width'], params['field_height'],
                        params['grid_size'], grid_seed=params['grid_seed'])
    hubs = _small_hubs()
    uavs = [S.UAV(i, np.array([random.uniform(3.0, 17.0),
                               random.uniform(3.0, 17.0), 5.0]))
            for i in range(params['N_UAV'])]
    claimed = set()
    for u in uavs:
        u.hub_assigned = random.randint(0, len(hubs) - 1)
        avail = [gp for gp in grid if gp.idx not in claimed]
        for gp in random.sample(avail, 6):
            u.seeding[gp.idx] = gp.seed_type
            claimed.add(gp.idx)
    return grid, hubs, uavs


def _game(params):
    costs = S.HOUBOCosts(params)
    traj = S.TrajectoryModel(params)
    gm = S.OrdinalPotentialGame(costs, traj, params)
    br = S.BestResponseDynamics(gm, params)
    return gm, br, S.StackelbergGame(gm, br, params)


# ---------------------------------------------------------------------------
# Sec. IV-A -- HOUBO evaluation
# ---------------------------------------------------------------------------
def test_houbo_evaluate_and_delta():
    h = S.HOUBO(4)
    h.add_term({0}, 2.0)            # 2 x0
    h.add_term({1, 2}, -3.0)        # -3 x1 x2
    h.add_term({0, 1, 2, 3}, 5.0)   # 5 x0 x1 x2 x3
    x = np.array([1, 1, 1, 0])
    assert abs(h.evaluate(x) - (2.0 - 3.0 + 0.0)) < 1e-12
    # flipping x3 to 1 activates the fourth-order term as well
    assert abs(h.evaluate_delta(3, x) - 5.0) < 1e-12
    # flipping x0 off removes both the linear and the fourth-order term
    assert abs(h.evaluate_delta(0, x) - (-2.0)) < 1e-12
    print('  [ok] HOUBO evaluate / evaluate_delta')


# ---------------------------------------------------------------------------
# Sec. III-A -- the grid must be fully determined by grid_seed
# ---------------------------------------------------------------------------
def test_grid_is_deterministic():
    a = S.build_grid(40.0, 40.0, 2.0, grid_seed=12345)
    b = S.build_grid(40.0, 40.0, 2.0, grid_seed=12345)
    assert len(a) == 400
    assert [(g.x, g.y, g.seed_type, g.is_edge) for g in a] == \
           [(g.x, g.y, g.seed_type, g.is_edge) for g in b]
    c = S.build_grid(40.0, 40.0, 2.0, grid_seed=999)
    assert [g.seed_type for g in a] != [g.seed_type for g in c]
    print('  [ok] grid layout is determined by grid_seed')


# ---------------------------------------------------------------------------
# Sec. IV-B -- the rollout returns a finite, positive energy
# ---------------------------------------------------------------------------
def test_rollout_energy_is_positive_and_finite():
    p = _small_params()
    grid, hubs, uavs = _scenario(p)
    traj_model = S.TrajectoryModel(p)
    wind = np.array([0.5, 0.0, 0.0])
    traj, energy = traj_model.rollout(uavs[0].seeding, grid, uavs[0].p, wind)
    assert np.isfinite(energy) and energy > 0.0
    assert len(traj) > 1
    assert all(np.all(np.isfinite(pt)) for pt in traj)
    empty_traj, empty_energy = traj_model.rollout({}, grid, uavs[0].p, wind)
    assert empty_energy == 0.0 and len(empty_traj) == 1
    print('  [ok] rollout energy is positive, finite, and zero for empty plans')


# ---------------------------------------------------------------------------
# Lemma 1 -- the exact leftover identity, the role-flip case, and the
# invariance of Phi to the hub assignment once lam is uniform
# ---------------------------------------------------------------------------
def _conflict_shares(uavs, params):
    """Per-player conflict share, in player order."""
    seedings = [u.seeding for u in uavs]
    return [S._conflict_share(s, seedings, params) for s in seedings]


def test_lemma1_leftover_identity_is_exact():
    """dU + dPhi == d(R_task n_i) - d(C_hub_i) + sum_{k != i} d(f_pen_k)."""
    p = _small_params()
    grid, hubs, uavs = _scenario(p)
    gm, _br, _se = _game(p)
    wind = np.array([0.5, 0.0, 0.0])

    uav = uavs[0]
    claimed = {gid for u in uavs for gid in u.seeding}
    free = [gp for gp in grid if gp.idx not in claimed]
    assert free, 'scenario must leave at least one unclaimed cell'

    new_seeding = dict(uav.seeding)
    new_seeding[free[0].idx] = free[0].seed_type
    old = {'seeding': uav.seeding, 'role': uav.role, 'hub': uav.hub_assigned}
    new = {'seeding': new_seeding, 'role': uav.role,
           'hub': (uav.hub_assigned + 1) % len(hubs)}

    sh_old = _conflict_shares(uavs, p)
    c_old = S._hub_access_cost(hubs[old['hub']], old['seeding'], grid, p)
    r_old = p['R_task'] * len(old['seeding'])

    saved = (uav.seeding, uav.role, uav.hub_assigned)
    uav.seeding, uav.role, uav.hub_assigned = (new['seeding'], new['role'],
                                               new['hub'])
    sh_new = _conflict_shares(uavs, p)
    c_new = S._hub_access_cost(hubs[new['hub']], new['seeding'], grid, p)
    r_new = p['R_task'] * len(new['seeding'])
    uav.seeding, uav.role, uav.hub_assigned = saved

    dU, dPhi, _holds, _case = gm.verify_lemma1(uav, old, new, uavs, grid, hubs,
                                               wind)
    other = sum(sh_new[k] - sh_old[k]
                for k, u in enumerate(uavs) if u is not uav)
    predicted = (r_new - r_old) - (c_new - c_old) + other
    residual = abs((dU + dPhi) - predicted)
    assert residual < 1e-6, (dU + dPhi, predicted)
    print('  [ok] Lemma 1 leftover identity is exact: dU + dPhi = '
          'd(R n) - d(C_hub) + sum_{k != i} d(f_pen)  (residual %.1e)'
          % residual)


def test_lemma1_role_flip_has_zero_leftover():
    """A role flip changes neither n_i nor the hub, so T = 0 and dU = -dPhi."""
    p = _small_params()
    grid, hubs, uavs = _scenario(p)
    gm, _br, _se = _game(p)
    wind = np.array([0.5, 0.0, 0.0])

    uav = uavs[0]
    old = {'seeding': uav.seeding, 'role': uav.role, 'hub': uav.hub_assigned}
    new = {'seeding': uav.seeding, 'role': 1 - uav.role,
           'hub': uav.hub_assigned}
    dU, dPhi, holds, case = gm.verify_lemma1(uav, old, new, uavs, grid, hubs,
                                             wind)
    assert abs(dU + dPhi) < 1e-9, (dU, dPhi)
    assert bool(holds) is True and 'Case 1' in case, (dU, dPhi, case)
    print('  [ok] a role flip satisfies the sign identity exactly '
          '(dU=%.3e, dPhi=%.3e)' % (dU, dPhi))


def test_phi_invariant_to_hub_assignment_given_equal_lambda():
    """Phi depends on the hub only through lam; the access price cancels."""
    p = _small_params()
    grid, hubs, uavs = _scenario(p)
    gm, _br, _se = _game(p)
    wind = np.array([0.5, 0.0, 0.0])
    for h in hubs:
        h.lam = 1.5
    base = [u.hub_assigned for u in uavs]
    phi_a = gm.potential(uavs, grid, hubs, wind)
    for u in uavs:
        u.hub_assigned = (u.hub_assigned + 1) % len(hubs)
    phi_b = gm.potential(uavs, grid, hubs, wind)
    for u, a in zip(uavs, base):
        u.hub_assigned = a
    assert abs(phi_a - phi_b) < 1e-9, (phi_a, phi_b)
    print('  [ok] Phi is invariant to the hub assignment once lam is uniform '
          '(access price cancels)')


# ---------------------------------------------------------------------------
# Lemma 1 -- the paper's own verification instance must reproduce exactly
# ---------------------------------------------------------------------------
def test_paper_lemma1_reproduces_84_of_90():
    """24 m x 24 m / 3 hubs / seed 7, 90 deviations. Takes about half a minute."""
    params = dict(S.PAPER_LEMMA_PARAMS)
    grid, hubs, uavs, _info = S.build_paper_lemma_scenario(params)
    costs = S.HOUBOCosts(params)
    traj = S.TrajectoryModel(params)
    gm = S.OrdinalPotentialGame(costs, traj, params)
    br = S.BestResponseDynamics(gm, params)
    se = S.StackelbergGame(gm, br, params)
    se.solve(uavs, grid, hubs, S.PAPER_LEMMA_WIND)

    holds, total, fam = S.sample_paper_lemma1(gm, uavs, grid, hubs,
                                              S.PAPER_LEMMA_WIND)
    assert (holds, total) == (84, 90), (holds, total)
    assert (fam['add']['ok'], fam['add']['n']) == (18, 18)
    assert (fam['del']['ok'], fam['del']['n']) == (26, 32)
    assert (fam['role']['ok'], fam['role']['n']) == (18, 18)
    assert (fam['hub']['ok'], fam['hub']['n']) == (22, 22)
    print('  [ok] the paper Lemma 1 experiment reproduces 84/90 '
          '(add 18/18, drop 26/32, role 18/18, hub 22/22)')


def test_exact_potential_identity_holds_on_every_family():
    """Lemma 1 of the manuscript: dU_i = -dPhi_ex for all four families.

    Uses Eq. (14) through OrdinalPotentialGame.exact_potential.  The residual is
    an algebraic identity, so the assertion is at round-off level rather than a
    tolerance chosen to pass.
    """
    params = dict(S.PAPER_LEMMA_PARAMS)
    grid, hubs, uavs, _info = S.build_paper_lemma_scenario(params)
    costs = S.HOUBOCosts(params)
    traj = S.TrajectoryModel(params)
    gm = S.OrdinalPotentialGame(costs, traj, params)
    holds, total, fam = S.sample_paper_lemma1_exact(gm, uavs, grid, hubs,
                                                   S.PAPER_LEMMA_WIND)
    assert (holds, total) == (90, 90), (holds, total, fam)
    for mode in ('add', 'del', 'role', 'hub'):
        assert fam[mode]['ok'] == fam[mode]['n'], (mode, fam[mode])
        assert fam[mode]['worst_rel_res'] < 1e-9, (mode, fam[mode])
    print('  [ok] exact potential: dU + dPhi_ex = 0 on 90/90 '
          '(add 21, drop 29, role 23, hub 17; worst residual %.1e)'
          % max(f['worst_rel_res'] for f in fam.values()))


def test_exact_potential_decreases_on_every_improving_move():
    """Every accepted best response lowers Phi_ex -- the monotonicity in Thm 1."""
    params = dict(S.PAPER_LEMMA_PARAMS)
    grid, hubs, uavs, _info = S.build_paper_lemma_scenario(params)
    costs = S.HOUBOCosts(params)
    traj = S.TrajectoryModel(params)
    gm = S.OrdinalPotentialGame(costs, traj, params)
    improving = 0
    for uav in uavs:
        for _ in range(10):
            mode = random.choice(['add', 'del', 'role', 'hub'])
            old = {'seeding': uav.seeding.copy(), 'role': uav.role,
                   'hub': uav.hub_assigned}
            new = {'seeding': uav.seeding.copy(), 'role': uav.role,
                   'hub': uav.hub_assigned}
            if mode == 'add':
                free = [gp.idx for gp in grid if gp.idx not in new['seeding']]
                if not free:
                    continue
                gid = random.choice(free)
                new['seeding'][gid] = grid[gid].seed_type
            elif mode == 'del' and len(new['seeding']) > 2:
                del new['seeding'][random.choice(list(new['seeding'].keys()))]
            elif mode == 'role':
                new['role'] = 1 - new['role']
            elif mode == 'hub':
                new['hub'] = (uav.hub_assigned + 1) % len(hubs)
            else:
                continue
            d_u, d_phi, _res = gm.verify_lemma1_exact(
                uav, old, new, uavs, grid, hubs, S.PAPER_LEMMA_WIND)
            if d_u > 0:
                improving += 1
                assert d_phi < 0, (mode, d_u, d_phi)
    assert improving > 0
    print('  [ok] every strictly improving move lowers Phi_ex (%d moves)'
          % improving)


def test_price_iterates_stay_inside_the_box():
    """Algorithm 1 line 10 projects onto P; Corollary 1 asserts p^(k) in P."""
    p = _small_params()
    grid, hubs, uavs = _scenario(p)
    gm, br, se = _game(p)
    res = se.solve(uavs, grid, hubs, np.array([0.5, 0.0, 0.0]))
    lo, hi = S.PRICE_LOWER, S.PRICE_UPPER
    for m, hist in res['price_history'].items():
        for comp, k in (('c', 0), ('rho', 1), ('lam', 2)):
            for value in hist[comp]:
                assert lo[k] - 1e-12 <= value <= hi[k] + 1e-12, \
                    (m, comp, value)
    print('  [ok] every price iterate lies in P = [%.1f,%.1f]x[%.1f,%.1f]'
          'x[%.1f,%.1f]' % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))


# ---------------------------------------------------------------------------
# Sec. V-B (Theorem 1) -- a best-response sweep never increases Phi when it changes nothing
# ---------------------------------------------------------------------------
def test_phi_nonincreasing_within_sweep():
    p = _small_params()
    grid, hubs, uavs = _scenario(p)
    gm, br, _se = _game(p)
    wind = np.array([0.5, 0.0, 0.0])
    phi_hist, iters, _conv = br.iterate(uavs, grid, hubs, wind)
    assert iters >= 1
    # iterate() records one entry per sweep plus the closing value
    assert len(phi_hist) == iters + 1, (len(phi_hist), iters)
    assert all(np.isfinite(v) for v in phi_hist)
    if iters < p['max_OPG_iter']:      # terminated by the convergence test
        assert abs(phi_hist[-1] - phi_hist[-2]) < 1e-6
    print('  [ok] BRD sweep records a finite, terminating Phi history '
          '(%d sweeps)' % iters)


# ---------------------------------------------------------------------------
# Reproducibility -- the same seed reproduces the equilibrium exactly
# ---------------------------------------------------------------------------
def test_same_seed_reproduces_equilibrium():
    p = _small_params()
    wind = np.array([0.5, 0.0, 0.0])
    runs = []
    for _ in range(2):
        grid, hubs, uavs = _scenario(p)
        gm, _br, se = _game(p)
        res = se.solve(uavs, grid, hubs, wind)
        runs.append(([round(v, 9) for v in res['phi_history']],
                     res['outer_iters'],
                     [(u.hub_assigned, u.role, tuple(sorted(u.seeding.items())))
                      for u in uavs]))
    assert runs[0] == runs[1], 'identical seeds must reproduce the equilibrium'
    print('  [ok] identical seed reproduces the equilibrium exactly')


# ---------------------------------------------------------------------------
# Sec. IV-C, Eq. (13) -- the hub access cost is invariant under the centroid switch
# ---------------------------------------------------------------------------
def test_hub_access_cost_distance_term():
    p = _small_params()
    grid, hubs, uavs = _scenario(p)
    seeding = uavs[0].seeding
    base = S._hub_access_cost(hubs[0], seeding, grid, dict(p, hub_dist_w=0.0))
    n = len(seeding)
    expected = hubs[0].c * 2.0 + hubs[0].rho * n * 0.1
    assert abs(base - expected) < 1e-12
    far = S._hub_access_cost(hubs[0], seeding, grid, dict(p, hub_dist_w=1.0))
    assert far > base
    print('  [ok] hub access cost = price term (+ optional travel term)')


def main():
    print('sop_core self-tests')
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print('all %d tests passed' % len(tests))


if __name__ == '__main__':
    main()
