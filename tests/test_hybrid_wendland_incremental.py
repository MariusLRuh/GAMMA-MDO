from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gamma_mdo.core.sdf.rbf_based.dwr_refinement import (
    HybridWendlandC2DWRRefinerIncremental,
    HybridWendlandC2DWRRefinerOptimized,
    assemble_wendland_matrix,
    evaluate_rbf_matvec,
    factorize_spd,
)


def _make_dataset():
    rng = np.random.default_rng(7)
    pts = rng.uniform(-1.0, 1.0, size=(96, 3))
    center = np.array([0.15, -0.2, 0.1])
    sdf = np.linalg.norm(pts - center[None, :], axis=1) - 0.6

    rank = np.argsort(np.abs(sdf))
    local_init = rank[:18]
    global_init = rank[18:30]
    return pts, sdf, local_init, global_init


def _make_far_field_dataset():
    center = np.array([0.15, -0.2, 0.1])
    box_min = np.array([-0.9, -0.95, -0.9], dtype=float)
    box_max = np.array([0.9, 0.95, 0.9], dtype=float)
    outer_min = np.array([-3.2, -3.0, -3.1], dtype=float)
    outer_max = np.array([3.2, 3.0, 3.1], dtype=float)
    spacing = np.array([0.8, 0.75, 0.85], dtype=float)

    axes = []
    for lo, hi, h in zip(outer_min, outer_max, spacing):
        coords = np.arange(lo, hi + 0.5 * h, h, dtype=float)
        coords[0] = lo
        if coords[-1] < hi - 1e-12:
            coords = np.append(coords, hi)
        else:
            coords[-1] = hi
        axes.append(coords)

    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    all_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    inside = np.all((all_points >= box_min[None, :]) & (all_points <= box_max[None, :]), axis=1)
    far_field_points = all_points[~inside]
    far_field_sdf = np.linalg.norm(far_field_points - center[None, :], axis=1) - 0.6
    return far_field_points, far_field_sdf, box_min, box_max


def _make_refiner(refiner_cls, use_far_field: bool = False):
    pts, sdf, local_init, global_init = _make_dataset()
    kwargs = {}
    if use_far_field:
        far_field_points, far_field_sdf, far_field_box_min, far_field_box_max = _make_far_field_dataset()
        kwargs.update(
            far_field_points=far_field_points,
            far_field_sdf=far_field_sdf,
            far_field_box_min=far_field_box_min,
            far_field_box_max=far_field_box_max,
            far_field_neighbor_k=4,
            far_field_radius_scale=1.75,
            far_field_radius_quantile=0.9,
            far_field_radius_inflate=1.05,
            far_field_ridge=1e-12,
            far_field_blend_inner_distance=0.4,
            far_field_blend_outer_distance=0.6,
        )

    return refiner_cls(
        all_points=pts,
        all_sdf=sdf,
        initial_local_train_indices=local_init,
        initial_global_train_indices=global_init,
        shortlist_size=14,
        shortlist_fraction=0.2,
        freeze_global=True,
        enable_adjoint=True,
        **kwargs,
    )


def _enable_small_incremental_batches(refiner):
    refiner._incremental_min_old_size = 10
    refiner._incremental_max_batch_size = 4
    refiner._incremental_max_relative_batch = 0.5
    refiner._local_rebuild_every = 8


def test_incremental_refiner_api_parity_against_optimized():
    pts, _, _, _ = _make_dataset()
    optimized = _make_refiner(HybridWendlandC2DWRRefinerOptimized)
    incremental = _make_refiner(HybridWendlandC2DWRRefinerIncremental)

    score_opt = optimized.compute_dwr_scores()
    score_inc = incremental.compute_dwr_scores()

    np.testing.assert_array_equal(score_inc.train_indices, score_opt.train_indices)
    np.testing.assert_array_equal(score_inc.check_indices, score_opt.check_indices)
    np.testing.assert_allclose(
        incremental.predict(pts[:12]),
        optimized.predict(pts[:12]),
        atol=1e-8,
        rtol=1e-6,
    )
    np.testing.assert_allclose(score_inc.eta, score_opt.eta, atol=1e-8, rtol=1e-6)

    ref_opt = _make_refiner(HybridWendlandC2DWRRefinerOptimized)
    ref_inc = _make_refiner(HybridWendlandC2DWRRefinerIncremental)
    res_opt = ref_opt.refine_once(
        n_add_local=2,
        min_separation_local=0.0,
        recompute_scores_after_update=True,
    )
    res_inc = ref_inc.refine_once(
        n_add_local=2,
        min_separation_local=0.0,
        recompute_scores_after_update=True,
    )
    np.testing.assert_array_equal(res_inc.selected_check_indices, res_opt.selected_check_indices)
    np.testing.assert_allclose(
        ref_inc.predict(pts[:12]),
        ref_opt.predict(pts[:12]),
        atol=1e-8,
        rtol=1e-6,
    )

    history = _make_refiner(HybridWendlandC2DWRRefinerIncremental).refine(
        n_iterations=2,
        n_add_local_per_iter=1,
        min_separation_local=0.0,
        verbose=False,
        recompute_scores_after_update=True,
    )
    assert len(history) == 2


def test_incremental_refiner_no_update_keeps_state_consistent():
    pts, _, _, _ = _make_dataset()
    refiner = _make_refiner(HybridWendlandC2DWRRefinerIncremental)
    baseline = refiner.compute_dwr_scores()
    train_before = refiner._local_state.train_indices.copy()
    pred_before = refiner.predict(pts)

    result = refiner.refine_once(
        n_add_local=0,
        min_separation_local=0.0,
        recompute_scores_after_update=True,
    )

    np.testing.assert_array_equal(refiner._local_state.train_indices, train_before)
    np.testing.assert_allclose(refiner.predict(pts), pred_before, atol=1e-10, rtol=1e-8)
    assert result.diagnostics["n_added_local"] == 0
    assert result.diagnostics["local_update_mode"] == "reuse"
    np.testing.assert_allclose(result.eta, baseline.eta, atol=1e-10, rtol=1e-8)


def test_incremental_local_update_matches_full_rebuild_reference():
    pts, _, _, _ = _make_dataset()
    refiner = _make_refiner(HybridWendlandC2DWRRefinerIncremental)
    _enable_small_incremental_batches(refiner)

    refiner.compute_dwr_scores()
    result = refiner.refine_once(
        n_add_local=1,
        min_separation_local=0.0,
        recompute_scores_after_update=True,
    )

    assert result.diagnostics["n_added_local"] == 1
    assert result.diagnostics["local_update_mode"] == "incremental"

    info = refiner.get_current_fit_info()
    local_idx = info["local_train_idx"]
    X_local = info["X_local"]
    rho_local_stage = info["rho_global"]
    local_tree = cKDTree(X_local)

    local_background = refiner._evaluate_global_model_on_indices(
        local_idx,
        info["X_global"],
        info["c_global"],
        info["global_rho"],
        info["global_tr_tree"],
    )
    y_local_residual = refiner.all_sdf[local_idx] - local_background
    K_local = assemble_wendland_matrix(
        X_local,
        X_local,
        rho_local_stage,
        src_tree=local_tree,
        chunk_size=refiner.assembly_chunk_size,
    ).tocsc()
    K_local = K_local + refiner.jitter * sparse.eye(K_local.shape[0], format="csc")
    coeffs_reference = np.asarray(factorize_spd(K_local)(y_local_residual), dtype=float).reshape(-1)

    global_pred = refiner._evaluate_global_model_on_indices(
        np.arange(pts.shape[0], dtype=int),
        info["X_global"],
        info["c_global"],
        info["global_rho"],
        info["global_tr_tree"],
    )
    local_pred_reference = evaluate_rbf_matvec(
        X_local,
        pts,
        rho_local_stage,
        coeffs_reference,
        src_tree=local_tree,
    )
    pred_reference = global_pred + local_pred_reference
    np.testing.assert_allclose(refiner.predict(pts), pred_reference, atol=1e-8, rtol=1e-6)


def test_incremental_fallback_rebuilds_cleanly():
    refiner = _make_refiner(HybridWendlandC2DWRRefinerIncremental)
    _enable_small_incremental_batches(refiner)
    refiner.compute_dwr_scores()
    refiner._debug_force_incremental_failure = True

    result = refiner.refine_once(
        n_add_local=1,
        min_separation_local=0.0,
        recompute_scores_after_update=True,
    )

    assert result.diagnostics["n_added_local"] == 1
    assert result.diagnostics["local_update_mode"] == "full"
    assert result.diagnostics["local_incremental_fallback"] is True


def test_stage_radius_stays_fixed_until_rebuild():
    refiner = _make_refiner(HybridWendlandC2DWRRefinerIncremental)
    _enable_small_incremental_batches(refiner)

    initial = refiner.compute_dwr_scores()
    rho_stage_0 = initial.diagnostics["rho_global"]

    incremental = refiner.refine_once(
        n_add_local=1,
        min_separation_local=0.0,
        recompute_scores_after_update=True,
    )
    assert incremental.diagnostics["local_update_mode"] == "incremental"
    assert incremental.diagnostics["local_stage_radius_recomputed"] is False
    assert np.isclose(incremental.diagnostics["rho_global"], rho_stage_0)

    refiner._force_local_rebuild_next = True
    rebuilt = refiner.compute_dwr_scores()
    assert rebuilt.diagnostics["local_update_mode"] == "full"
    assert rebuilt.diagnostics["local_stage_radius_recomputed"] is True


def test_three_stage_optimized_fit_matches_reference_sequential_solves():
    pts, _, _, _ = _make_dataset()
    refiner = _make_refiner(HybridWendlandC2DWRRefinerOptimized, use_far_field=True)
    info = refiner.get_current_fit_info(force_refit=True)

    rho_ff = info["far_field_rho"]
    X_ff = info["X_far_field"]
    y_ff = info["y_far_field"]
    far_tree = cKDTree(X_ff)
    K_ff = assemble_wendland_matrix(
        X_ff,
        X_ff,
        rho_ff,
        src_tree=far_tree,
        chunk_size=refiner.assembly_chunk_size,
    ).tocsc()
    K_ff = K_ff + refiner.far_field_ridge * sparse.eye(K_ff.shape[0], format="csc")
    coeffs_ff_ref = np.asarray(factorize_spd(K_ff)(y_ff), dtype=float).reshape(-1)

    X_global = info["X_global"]
    global_tree = cKDTree(X_global)
    far_pred_global = refiner._evaluate_far_field_values(
        X_global,
        X_ff,
        coeffs_ff_ref,
        rho_ff,
        info["far_field_blend_inner_radius"],
        info["far_field_blend_outer_radius"],
    )
    y_global_residual = info["y_global"] - far_pred_global
    K_global = assemble_wendland_matrix(
        X_global,
        X_global,
        info["global_rho"],
        src_tree=global_tree,
        chunk_size=refiner.assembly_chunk_size,
    ).tocsc()
    K_global = K_global + refiner.global_ridge * sparse.eye(K_global.shape[0], format="csc")
    coeffs_global_ref = np.asarray(factorize_spd(K_global)(y_global_residual), dtype=float).reshape(-1)

    X_local = info["X_local"]
    local_tree = cKDTree(X_local)
    far_pred_local = refiner._evaluate_far_field_values(
        X_local,
        X_ff,
        coeffs_ff_ref,
        rho_ff,
        info["far_field_blend_inner_radius"],
        info["far_field_blend_outer_radius"],
    )
    global_pred_local = evaluate_rbf_matvec(
        X_global,
        X_local,
        info["global_rho"],
        coeffs_global_ref,
        src_tree=global_tree,
    )
    y_local_residual = info["y_local"] - far_pred_local - global_pred_local
    K_local = assemble_wendland_matrix(
        X_local,
        X_local,
        info["rho_global"],
        src_tree=local_tree,
        chunk_size=refiner.assembly_chunk_size,
    ).tocsc()
    K_local = K_local + refiner.jitter * sparse.eye(K_local.shape[0], format="csc")
    coeffs_local_ref = np.asarray(factorize_spd(K_local)(y_local_residual), dtype=float).reshape(-1)

    np.testing.assert_allclose(info["c_far_field"], coeffs_ff_ref, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(info["c_global"], coeffs_global_ref, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(info["c_local"], coeffs_local_ref, atol=1e-10, rtol=1e-8)

    far_pred_all = refiner._evaluate_far_field_values(
        pts,
        X_ff,
        coeffs_ff_ref,
        rho_ff,
        info["far_field_blend_inner_radius"],
        info["far_field_blend_outer_radius"],
    )
    global_pred_all = evaluate_rbf_matvec(
        X_global,
        pts,
        info["global_rho"],
        coeffs_global_ref,
        src_tree=global_tree,
    )
    local_pred_all = evaluate_rbf_matvec(
        X_local,
        pts,
        info["rho_global"],
        coeffs_local_ref,
        src_tree=local_tree,
    )
    pred_reference = far_pred_all + global_pred_all + local_pred_all
    np.testing.assert_allclose(refiner.predict(pts), pred_reference, atol=1e-8, rtol=1e-6)


def test_blended_far_field_stage_is_suppressed_in_central_region():
    refiner = _make_refiner(HybridWendlandC2DWRRefinerOptimized, use_far_field=True)
    info = refiner.get_current_fit_info(force_refit=True)
    center_point = np.array([[0.15, -0.2, 0.1]])

    raw = evaluate_rbf_matvec(
        info["X_far_field"],
        center_point,
        info["far_field_rho"],
        info["c_far_field"],
        src_tree=cKDTree(info["X_far_field"]),
    )[0]
    blended = refiner._evaluate_far_field_values(
        center_point,
        info["X_far_field"],
        info["c_far_field"],
        info["far_field_rho"],
        info["far_field_blend_inner_radius"],
        info["far_field_blend_outer_radius"],
    )[0]

    assert abs(raw) > 1e-10
    assert np.isclose(blended, 0.0)


def test_incremental_refiner_matches_optimized_with_far_field_stage():
    pts, _, _, _ = _make_dataset()
    optimized = _make_refiner(HybridWendlandC2DWRRefinerOptimized, use_far_field=True)
    incremental = _make_refiner(HybridWendlandC2DWRRefinerIncremental, use_far_field=True)

    score_opt = optimized.compute_dwr_scores()
    score_inc = incremental.compute_dwr_scores()

    np.testing.assert_array_equal(score_inc.train_indices, score_opt.train_indices)
    np.testing.assert_array_equal(score_inc.check_indices, score_opt.check_indices)
    np.testing.assert_allclose(
        incremental.predict(pts[:12]),
        optimized.predict(pts[:12]),
        atol=1e-8,
        rtol=1e-6,
    )
    np.testing.assert_allclose(score_inc.eta, score_opt.eta, atol=1e-8, rtol=1e-6)
    assert score_inc.diagnostics["n_far_field_train"] == score_opt.diagnostics["n_far_field_train"]
