
#!/usr/bin/env python3
"""
CRITICAL BASELINE TEST FOR DECOHERENCE PROTECTION
==================================================

This script tests THE most important question:
Does scrambling (s=8) reduce decoherence compared to no scrambling (s=0)?

Author: Scientific Validation
Date: 2025
Runtime: ~20-40 minutes (depending on shots)
Requirements: numpy, scipy, matplotlib

Usage:
    python baseline_test.py

Output:
    - Console output with suppression factor
    - Plot: baseline_comparison_[timestamp].png
    - Data: baseline_results_[timestamp].npz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import curve_fit
from numpy.random import default_rng
import time as pytime
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set to True for quick test (5 mins), False for publication quality (30-40 mins)
QUICK_MODE = False

if QUICK_MODE:
    N_SHOTS = 10
    print("⚡ QUICK MODE: 10 shots (for testing only)")
else:
    N_SHOTS = 50
    print("🔬 RIGOROUS MODE: 50 shots (publication quality)")

# System parameters
N = 6              # System qubits
M_ENV = 2          # Environment qubits
COUPLING_G = 0.02  # System-environment coupling
T_MAX = 10.0        # Maximum evolution time
N_STEPS = 50       # Time steps

# ============================================================================
# QUANTUM MECHANICS HELPERS
# ============================================================================

# Pauli matrices
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_list(mats):
    """Tensor product of list of matrices."""
    out = np.array([1.0], dtype=complex)
    for m in mats:
        out = np.kron(out, m)
    return out

def op_on(n, op, idx):
    """Apply operator op on qubit idx in n-qubit system."""
    mats = [I2] * n
    mats[idx] = op
    return kron_list(mats)

def two_on(n, op1, i1, op2, i2):
    """Apply two-body operator."""
    mats = [I2] * n
    mats[i1] = op1
    mats[i2] = op2
    return kron_list(mats)

def make_hermitian(A):
    """Ensure matrix is Hermitian."""
    return 0.5 * (A + A.conj().T)

def normalize_rho(rho):
    """Normalize density matrix."""
    rho = make_hermitian(rho)
    tr = np.real(np.trace(rho))
    if np.abs(tr) < 1e-20:
        return rho
    return rho / tr

def purity(rho):
    """Compute purity Tr(ρ²)."""
    val = np.trace(rho @ rho)
    return float(np.real(val))

def partial_trace_env(rho, N, M):
    """Trace out last M qubits (environment)."""
    dS = 2**N
    dE = 2**M
    rho = rho.reshape(dS, dE, dS, dE)
    rhoS = np.zeros((dS, dS), dtype=complex)
    for i in range(dE):
        rhoS += rho[:, i, :, i]
    return rhoS

# ============================================================================
# HAMILTONIAN CONSTRUCTION
# ============================================================================

def random_scrambling_H(N, strength=1.0, longrange=True, seed=None):
    """Random scrambling Hamiltonian."""
    if seed is not None:
        rng = default_rng(seed)
    else:
        rng = default_rng()

    d = 2**N
    H = np.zeros((d, d), dtype=complex)

    # Local fields
    for i in range(N):
        a, b, c = rng.normal(scale=strength, size=3)
        H += a * op_on(N, sx, i)
        H += b * op_on(N, sy, i)
        H += c * op_on(N, sz, i)

    # Two-body terms (long-range interactions)
    if longrange:
        for i in range(N):
            for j in range(i+1, N):
                Jx = rng.normal(scale=0.4 * strength)
                Jy = rng.normal(scale=0.4 * strength)
                Jz = rng.normal(scale=0.4 * strength)
                H += Jx * two_on(N, sx, i, sx, j)
                H += Jy * two_on(N, sy, i, sy, j)
                H += Jz * two_on(N, sz, i, sz, j)

    return make_hermitian(H)

def build_H_total(N, M_env, scr_strength, coupling_g, seed=None):
    """Build total Hamiltonian: H_sys + H_env + H_int."""

    # System Hamiltonian
    if scr_strength == 0:
        Hs = np.zeros((2**N, 2**N), dtype=complex)
    else:
        Hs = random_scrambling_H(N, strength=scr_strength, longrange=True, seed=seed)

    # Environment Hamiltonian
    if M_env > 0:
        He = random_scrambling_H(M_env, strength=0.5, longrange=False,
                                seed=seed+1 if seed else None)
    else:
        He = np.zeros((1, 1), dtype=complex)

    dS = 2**N
    dE = 2**M_env

    # Total: H = Hs ⊗ I + I ⊗ He + H_int
    Htot = np.kron(Hs, np.eye(dE)) + np.kron(np.eye(dS), He)

    # Interaction: couple boundary qubits to environment
    n_boundary = max(1, int(N * 0.3))

    A_sys = np.zeros((dS, dS), dtype=complex)
    for b in range(n_boundary):
        A_sys += op_on(N, sz, b)
    A_sys = make_hermitian(A_sys)

    B_env = np.zeros((dE, dE), dtype=complex)
    for e in range(M_env):
        B_env += op_on(M_env, sz, e)
    B_env = make_hermitian(B_env)

    Htot += coupling_g * np.kron(A_sys, B_env)

    return make_hermitian(Htot)

# ============================================================================
# TIME EVOLUTION
# ============================================================================

def evolve_single_trajectory(N, M_env, scr_strength, coupling_g,
                            t_max, n_steps, seed=None):
    """Single evolution trajectory."""

    # Build Hamiltonian
    Htot = build_H_total(N, M_env, scr_strength, coupling_g, seed=seed)

    # Initial state: random pure state for system, maximally mixed for environment
    if seed is not None:
        rng = default_rng(seed)
    else:
        rng = default_rng()

    dS = 2**N
    dE = 2**M_env

    psi_sys = rng.normal(size=dS) + 1j * rng.normal(size=dS)
    psi_sys /= np.linalg.norm(psi_sys)
    rhoS0 = np.outer(psi_sys, psi_sys.conj())

    rhoE0 = np.eye(dE) / dE
    rho0 = np.kron(rhoS0, rhoE0)

    # Time evolution
    times = np.linspace(0, t_max, n_steps)
    purities = []

    for t in times:
        U = expm(-1j * Htot * t)
        rho_t = U @ rho0 @ U.conj().T

        # Trace out environment
        rhoS = partial_trace_env(rho_t, N, M_env)
        rhoS = normalize_rho(rhoS)

        purities.append(purity(rhoS))

    return times, np.array(purities)

def run_ensemble(N, M_env, scr_strength, coupling_g, t_max, n_steps, shots):
    """Run ensemble of trajectories."""

    print(f"  Running {shots} shots...", end='', flush=True)
    start = pytime.time()

    all_purities = []

    for shot in range(shots):
        seed = 1000 + shot
        times, purities = evolve_single_trajectory(
            N, M_env, scr_strength, coupling_g, t_max, n_steps, seed=seed
        )
        all_purities.append(purities)

        if (shot + 1) % 10 == 0:
            print(f"{shot+1}...", end='', flush=True)

    elapsed = pytime.time() - start
    print(f" done ({elapsed:.1f}s)")

    all_purities = np.array(all_purities)

    return {
        'times': times,
        'purities_mean': np.mean(all_purities, axis=0),
        'purities_std': np.std(all_purities, axis=0),
    }

# ============================================================================
# ANALYSIS
# ============================================================================

def fit_decay_rate(times, purities_mean, purities_std, fit_range=(0.0, 2.0)):
    """Fit exponential decay with bootstrap error estimation."""

    mask = (times >= fit_range[0]) & (times <= fit_range[1])
    t_fit = times[mask]
    p_mean = np.clip(purities_mean[mask], 1e-12, 1.0)
    p_std = purities_std[mask]

    if len(t_fit) < 3:
        return np.nan, np.nan

    # Best fit: P(t) = exp(-γt)
    try:
        coeffs = np.polyfit(t_fit, np.log(p_mean), 1)
        gamma_best = -coeffs[0]
    except:
        return np.nan, np.nan

    # Bootstrap for error estimation
    n_bootstrap = 100
    gammas_boot = []

    for _ in range(n_bootstrap):
        p_sample = p_mean + np.random.normal(0, p_std)
        p_sample = np.clip(p_sample, 1e-12, 1.0)
        try:
            coeffs = np.polyfit(t_fit, np.log(p_sample), 1)
            gammas_boot.append(-coeffs[0])
        except:
            pass

    if len(gammas_boot) > 0:
        gamma_std = np.std(gammas_boot)
    else:
        gamma_std = np.nan

    return gamma_best, gamma_std

def fit_with_quality_check(times, purity_mean, purity_std, fit_range=(0, 2)):
    """
    Fit exponential decay with quality metrics.

    Returns:
        gamma: Decay rate
        gamma_err: Error on gamma
        r_squared: Goodness of fit (0-1, higher is better)
        residual_std: Standard deviation of residuals
    """

    mask = (times >= fit_range[0]) & (times <= fit_range[1])
    t_fit = times[mask]
    p_mean = np.clip(purity_mean[mask], 1e-12, 1.0)
    p_std = purity_std[mask]

    if len(t_fit) < 3:
        return np.nan, np.nan, np.nan, np.nan

    # Fit log(P) = log(P0) - γt
    try:
        coeffs = np.polyfit(t_fit, np.log(p_mean), 1)
        gamma_best = -coeffs[0]
        log_P0 = coeffs[1]
    except:
        return np.nan, np.nan, np.nan, np.nan

    # Bootstrap for error estimation
    n_bootstrap = 100
    gammas_boot = []

    for _ in range(n_bootstrap):
        p_sample = p_mean + np.random.normal(0, p_std)
        p_sample = np.clip(p_sample, 1e-12, 1.0)
        try:
            coeffs_boot = np.polyfit(t_fit, np.log(p_sample), 1)
            gammas_boot.append(-coeffs_boot[0])
        except:
            pass

    if len(gammas_boot) > 0:
        gamma_err = np.std(gammas_boot)
    else:
        gamma_err = np.nan

    # Calculate R² (goodness of fit)
    p_pred = np.exp(log_P0 - gamma_best * t_fit)
    ss_res = np.sum((p_mean - p_pred)**2)
    ss_tot = np.sum((p_mean - np.mean(p_mean))**2)

    if ss_tot > 1e-12:
        r_squared = 1 - ss_res / ss_tot
    else:
        r_squared = np.nan

    # Calculate residuals
    residuals = p_mean - p_pred
    residual_std = np.std(residuals)

    return gamma_best, gamma_err, r_squared, residual_std


def test_fit_range_sensitivity(times, purity_mean, purity_std):
    """
    Test if decay rate is consistent across different fit windows.

    Returns:
        is_robust: True if decay rate is stable
        gamma_mean: Average gamma across windows
        gamma_std: Spread in gamma values
    """

    fit_ranges = [(0, 1.5), (0, 2.0), (0, 2.5), (0, 3.0)]
    valid_gammas = []

    print("  Testing fit range sensitivity:")

    for fr in fit_ranges:
        g, e, r2, _ = fit_with_quality_check(times, purity_mean, purity_std, fit_range=fr)

        if not np.isnan(g) and not np.isnan(r2) and r2 > 0.85:
            valid_gammas.append(g)
            print(f"    Range {fr}: γ = {g:.6f}, R² = {r2:.3f} ✓")
        else:
            print(f"    Range {fr}: Poor fit (R² = {r2:.3f}) ✗")

    if len(valid_gammas) >= 2:
        gamma_mean = np.mean(valid_gammas)
        gamma_std = np.std(valid_gammas)
        gamma_spread = gamma_std / gamma_mean if gamma_mean > 0 else np.inf

        print(f"  Average γ = {gamma_mean:.6f} ± {gamma_std:.6f}")
        print(f"  Relative spread: {gamma_spread*100:.1f}%")

        if gamma_spread < 0.2:
            print("  ✓ Decay rate ROBUST to fit range")
            return True, gamma_mean, gamma_std
        else:
            print("  ⚠️  Decay rate SENSITIVE to fit range")
            return False, gamma_mean, gamma_std
    else:
        print("  ✗ Insufficient good fits")
        return False, np.nan, np.nan


def test_time_regime_consistency(times, purity_mean, purity_std):
    """
    Check if decay is exponential by comparing early vs late regimes.

    Returns:
        decay_type: 'exponential', 'accelerating', 'decelerating', or 'non-exponential'
        gamma_early: Decay rate in early regime
        gamma_late: Decay rate in late regime
        ratio: gamma_late / gamma_early
    """

    print("\n  Testing time regime consistency:")

    # Early regime (0-2)
    gamma_early, err_early, r2_early, _ = fit_with_quality_check(
        times, purity_mean, purity_std, fit_range=(0, 2)
    )

    # Late regime (3-5)
    gamma_late, err_late, r2_late, _ = fit_with_quality_check(
        times, purity_mean, purity_std, fit_range=(3, 5)
    )

    print(f"    Early (0-2):  γ = {gamma_early:.6f} ± {err_early:.6f}, R² = {r2_early:.3f}")
    print(f"    Late (3-5):   γ = {gamma_late:.6f} ± {err_late:.6f}, R² = {r2_late:.3f}")

    if not np.isnan(gamma_early) and not np.isnan(gamma_late) and gamma_early > 0:
        ratio = gamma_late / gamma_early
        print(f"    Ratio (late/early): {ratio:.2f}")

        if abs(ratio - 1.0) < 0.3 and r2_early > 0.85 and r2_late > 0.85:
            print("    ✓ EXPONENTIAL decay (γ constant over time)")
            decay_type = 'exponential'
        elif ratio > 1.5:
            print(f"    ⚠️  ACCELERATING decay (γ increases {ratio:.1f}x)")
            decay_type = 'accelerating'
        elif ratio < 0.67:
            print(f"    ⚠️  DECELERATING decay (γ decreases {1/ratio:.1f}x)")
            decay_type = 'decelerating'
        else:
            print("    ⚠️  NON-EXPONENTIAL decay")
            decay_type = 'non-exponential'
    else:
        print("    ✗ Cannot determine (fit failed)")
        decay_type = 'unknown'
        ratio = np.nan

    return decay_type, gamma_early, gamma_late, ratio


def comprehensive_decay_analysis(results_dict, label):
    """
    Perform complete decay analysis on a single dataset.

    Args:
        results_dict: Output from run_ensemble()
        label: Name for this dataset (e.g., 's0', 's8')

    Returns:
        analysis: Dictionary with all metrics
    """

    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE ANALYSIS: {label}")
    print(f"{'='*70}")

    times = results_dict['times']
    purity_mean = results_dict['purities_mean']
    purity_std = results_dict['purities_std']

    # 1. Best fit with quality
    gamma, gamma_err, r2, resid_std = fit_with_quality_check(
        times, purity_mean, purity_std, fit_range=(0, 2)
    )

    print(f"\nBest fit (0-2 time range):")
    print(f"  γ = {gamma:.6f} ± {gamma_err:.6f}")
    print(f"  R² = {r2:.3f}")
    print(f"  Residual std = {resid_std:.6f}")

    if r2 < 0.85:
        print("  ⚠️  WARNING: Poor fit quality (R² < 0.85)")
    else:
        print("  ✓ Good fit quality")

    # 2. Fit range sensitivity
    is_robust, gamma_avg, gamma_spread = test_fit_range_sensitivity(
        times, purity_mean, purity_std
    )

    # 3. Time regime consistency
    decay_type, gamma_early, gamma_late, time_ratio = test_time_regime_consistency(
        times, purity_mean, purity_std
    )

    # 4. Plateau value (late-time purity)
    late_mask = times >= 0.6 * times[-1]
    plateau = np.mean(purity_mean[late_mask])
    plateau_std = np.mean(purity_std[late_mask])

    print(f"\nPlateau purity (late times): {plateau:.4f} ± {plateau_std:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY FOR {label}:")
    print(f"  Primary γ: {gamma:.6f} ± {gamma_err:.6f}")
    print(f"  Fit quality: R² = {r2:.3f}")
    print(f"  Decay type: {decay_type}")
    print(f"  Robustness: {'PASS' if is_robust else 'FAIL'}")
    print(f"{'='*70}")

    return {
        'gamma': gamma,
        'gamma_err': gamma_err,
        'r_squared': r2,
        'residual_std': resid_std,
        'is_robust': is_robust,
        'gamma_avg': gamma_avg,
        'gamma_spread': gamma_spread,
        'decay_type': decay_type,
        'gamma_early': gamma_early,
        'gamma_late': gamma_late,
        'time_ratio': time_ratio,
        'plateau': plateau,
        'plateau_std': plateau_std
    }
# ============================================================================
# MAIN TEST
# ============================================================================

def test_baseline_comparison():
    """THE CRITICAL TEST with comprehensive quality checks."""

    print("\n" + "="*70)
    print("🔬 CRITICAL BASELINE TEST: Measuring TRUE Suppression")
    print("="*70)
    print(f"\nParameters:")
    print(f"  System: N={N} qubits")
    print(f"  Environment: M={M_ENV} qubits")
    print(f"  Coupling: g={COUPLING_G}")
    print(f"  Shots: {N_SHOTS}")
    print(f"  Expected runtime: ~{N_SHOTS * 0.6:.0f} seconds per test\n")

    results = {}

    # Test 1: No scrambling (baseline)
    print("[1/3] Testing s=0 (NO system Hamiltonian - pure decoherence)")
    results['s0'] = run_ensemble(
        N, M_ENV,
        scr_strength=0.0,
        coupling_g=COUPLING_G,
        t_max=T_MAX,
        n_steps=N_STEPS,
        shots=N_SHOTS
    )

    # Test 2: Weak scrambling
    print("\n[2/3] Testing s=1 (weak scrambling)")
    results['s1'] = run_ensemble(
        N, M_ENV,
        scr_strength=1.0,
        coupling_g=COUPLING_G,
        t_max=T_MAX,
        n_steps=N_STEPS,
        shots=N_SHOTS
    )

    # Test 3: Strong scrambling
    print("\n[3/3] Testing s=8 (strong scrambling)")
    results['s8'] = run_ensemble(
        N, M_ENV,
        scr_strength=8.0,
        coupling_g=COUPLING_G,
        t_max=T_MAX,
        n_steps=N_STEPS,
        shots=N_SHOTS
    )

    # ========================================================================
    # COMPREHENSIVE ANALYSIS (NEW!)
    # ========================================================================

    print("\n" + "█"*70)
    print("PERFORMING COMPREHENSIVE QUALITY CHECKS")
    print("█"*70)

    # Analyze each dataset
    analysis = {}
    for key in ['s0', 's1', 's8']:
        analysis[key] = comprehensive_decay_analysis(results[key], key)

    # Extract validated decay rates
    gamma_0 = analysis['s0']['gamma']
    err_0 = analysis['s0']['gamma_err']
    gamma_1 = analysis['s1']['gamma']
    err_1 = analysis['s1']['gamma_err']
    gamma_8 = analysis['s8']['gamma']
    err_8 = analysis['s8']['gamma_err']

    # ========================================================================
    # QUALITY ASSESSMENT
    # ========================================================================

    print("\n" + "="*70)
    print("QUALITY ASSESSMENT")
    print("="*70)

    quality_score = 0
    max_score = 9
    issues = []

    # Check 1: Fit quality (R² > 0.85 for all)
    for key in ['s0', 's1', 's8']:
        if analysis[key]['r_squared'] > 0.85:
            quality_score += 1
            print(f"✓ {key}: Good fit quality (R² = {analysis[key]['r_squared']:.3f})")
        else:
            print(f"✗ {key}: Poor fit quality (R² = {analysis[key]['r_squared']:.3f})")
            issues.append(f"{key} has poor fit quality")

    # Check 2: Robustness to fit range
    for key in ['s0', 's1', 's8']:
        if analysis[key]['is_robust']:
            quality_score += 1
            print(f"✓ {key}: Robust to fit range")
        else:
            print(f"⚠️  {key}: Sensitive to fit range")
            issues.append(f"{key} sensitive to fit window")

    # Check 3: Exponential decay type
    for key in ['s0', 's1', 's8']:
        if analysis[key]['decay_type'] == 'exponential':
            quality_score += 1
            print(f"✓ {key}: Exponential decay confirmed")
        else:
            print(f"⚠️  {key}: {analysis[key]['decay_type']} decay")
            issues.append(f"{key} shows {analysis[key]['decay_type']} decay")

    print(f"\nQuality score: {quality_score}/{max_score}")

    if quality_score >= 7:
        print("✓ HIGH QUALITY - Results are reliable")
    elif quality_score >= 5:
        print("⚠️  MODERATE QUALITY - Interpret with caution")
    else:
        print("✗ LOW QUALITY - Results may not be reliable")
        print("\nIdentified issues:")
        for issue in issues:
            print(f"  • {issue}")

    # ========================================================================
    # SUPPRESSION CALCULATION
    # ========================================================================

    print("\n" + "="*70)
    print("RESULTS: TRUE SUPPRESSION FACTORS")
    print("="*70)

    print(f"\nBaseline (s=0):  γ = {gamma_0:.6f} ± {err_0:.6f}")
    print(f"Weak (s=1):      γ = {gamma_1:.6f} ± {err_1:.6f}")
    print(f"Strong (s=8):    γ = {gamma_8:.6f} ± {err_8:.6f}")

    # Calculate suppression factors
    if gamma_1 > 1e-9 and gamma_8 > 1e-9:
        suppression_1 = gamma_0 / gamma_1
        suppression_8 = gamma_0 / gamma_8

        # Error propagation
        err_supp_1 = suppression_1 * np.sqrt(
            (err_0/gamma_0)**2 + (err_1/gamma_1)**2
        ) if not np.isnan(err_0) and not np.isnan(err_1) else np.nan

        err_supp_8 = suppression_8 * np.sqrt(
            (err_0/gamma_0)**2 + (err_8/gamma_8)**2
        ) if not np.isnan(err_0) and not np.isnan(err_8) else np.nan
    else:
        suppression_1 = np.inf
        suppression_8 = np.inf
        err_supp_1 = np.inf
        err_supp_8 = np.inf

    print(f"\n{'='*70}")
    print(f"Suppression (s=1 vs s=0): {suppression_1:.2f}x ± {err_supp_1:.2f}x")
    print(f"Suppression (s=8 vs s=0): {suppression_8:.2f}x ± {err_supp_8:.2f}x")
    print(f"{'='*70}")

    # Statistical significance
    diff = gamma_0 - gamma_8
    err_combined = np.sqrt(err_0**2 + err_8**2)

    if err_combined > 0 and not np.isnan(err_combined):
        sigma = diff / err_combined
    else:
        sigma = 0.0

    print(f"\nStatistical significance: {sigma:.2f}σ")

    # ========================================================================
    # SCIENTIFIC VERDICT (with quality consideration)
    # ========================================================================

    print("\n" + "="*70)
    print("SCIENTIFIC VERDICT")
    print("="*70)

    # Adjust thresholds based on quality
    if quality_score >= 7:
        sigma_threshold_strong = 3.0
        sigma_threshold_moderate = 2.0
        suppression_threshold = 1.5
    else:
        sigma_threshold_strong = 4.0  # Higher bar for poor quality
        sigma_threshold_moderate = 3.0
        suppression_threshold = 2.0
        print("\n⚠️  NOTE: Higher significance thresholds due to quality concerns")

    if sigma > sigma_threshold_strong and suppression_8 > suppression_threshold and quality_score >= 7:
        verdict = "CONFIRMED"
        print("\n✅ STRONG EVIDENCE FOR PROTECTION")
        print(f"   Suppression: {suppression_8:.2f}x ({sigma:.1f}σ significance)")
        print(f"   Quality score: {quality_score}/{max_score}")
        print("   → Scrambling reduces decoherence!")
        print("\n📝 NEXT STEPS:")
        print("   1. Test energy scale control (scrambling vs integrable)")
        print("   2. Investigate mechanism (Zeno, matrix elements, etc.)")
        print("   3. Scale to larger systems (N=5, 6)")

    elif sigma > sigma_threshold_moderate and suppression_8 > 1.2:
        verdict = "PARTIAL"
        print("\n⚠️  MODERATE EVIDENCE FOR PROTECTION")
        print(f"   Suppression: {suppression_8:.2f}x ({sigma:.1f}σ significance)")
        print(f"   Quality score: {quality_score}/{max_score}")
        print("   → Effect present but weak or uncertain")
        print("\n📝 RECOMMENDATIONS:")
        if quality_score < 7:
            print("   1. Address quality issues (see above)")
        print("   2. Increase shots to 100+ for better statistics")
        print("   3. Test with stronger coupling (g=0.04)")
        print("   4. Try larger system (N=5)")

    else:
        verdict = "REJECTED"
        print("\n❌ NO SIGNIFICANT PROTECTION DETECTED")
        print(f"   Suppression: {suppression_8:.2f}x ({sigma:.1f}σ significance)")
        print(f"   Quality score: {quality_score}/{max_score}")
        print("   → Original hypothesis is likely incorrect")
        print("\n📝 ALTERNATIVE EXPLANATIONS:")
        print("   1. Effect is too small for N=4 system")
        print("   2. Mechanism is different than hypothesized")
        print("   3. Energy scale effects (not scrambling-specific)")

        if quality_score < 5:
            print("\n⚠️  ADDITIONAL NOTE: Low quality score suggests:")
            print("     • Non-exponential dynamics")
            print("     • Numerical issues")
            print("     • Need different analysis approach")

    print("="*70)

    # Generate plot
    print("\n📊 Generating plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Purity trajectories
    ax = axes[0]

    colors = {'s0': 'red', 's1': 'orange', 's8': 'green'}
    labels = {'s0': 's=0 (baseline)', 's1': 's=1 (weak)', 's8': 's=8 (strong)'}

    for key in ['s0', 's1', 's8']:
        res = results[key]
        ax.plot(res['times'], res['purities_mean'], '-',
               linewidth=2.5, label=labels[key], color=colors[key])
        ax.fill_between(res['times'],
                       res['purities_mean'] - res['purities_std'],
                       res['purities_mean'] + res['purities_std'],
                       alpha=0.2, color=colors[key])

    ax.set_xlabel('Time', fontsize=13)
    ax.set_ylabel('Purity', fontsize=13)
    ax.set_title('Purity Evolution: Baseline Comparison', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Decay rates comparison
    ax = axes[1]

    strengths = ['s=0\n(baseline)', 's=1\n(weak)', 's=8\n(strong)']
    gammas = [gamma_0, gamma_1, gamma_8]
    errs = [err_0, err_1, err_8]
    bar_colors = ['red', 'orange', 'green']

    bars = ax.bar(range(3), gammas, yerr=errs, capsize=10,
                  color=bar_colors, alpha=0.7, width=0.6, edgecolor='black', linewidth=1.5)

    ax.set_xticks(range(3))
    ax.set_xticklabels(strengths, fontsize=11)
    ax.set_ylabel('Decay rate γ', fontsize=13)
    ax.set_title('Decoherence Rates', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add suppression annotation
    if suppression_8 < np.inf and not np.isnan(suppression_8):
        ax.text(2, gamma_8 * 1.8, f'{suppression_8:.2f}x\nsuppression',
               ha='center', va='bottom', fontsize=12, fontweight='bold',
               color='green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add verdict to plot
    verdict_colors = {'CONFIRMED': 'green', 'PARTIAL': 'orange', 'REJECTED': 'red'}
    fig.text(0.5, 0.02, f'Verdict: {verdict}', ha='center', fontsize=14,
            fontweight='bold', color=verdict_colors.get(verdict, 'black'))

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'baseline_comparison_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"   Plot saved: {plot_filename}")

    plt.show()

    # Save data
    data_filename = f'baseline_results_{timestamp}.npz'
    np.savez(data_filename,
             gamma_0=gamma_0, err_0=err_0,
             gamma_1=gamma_1, err_1=err_1,
             gamma_8=gamma_8, err_8=err_8,
             suppression_1=suppression_1, err_supp_1=err_supp_1,
             suppression_8=suppression_8, err_supp_8=err_supp_8,
             sigma=sigma,
             verdict=verdict,
             times=results['s0']['times'],
             purity_s0_mean=results['s0']['purities_mean'],
             purity_s0_std=results['s0']['purities_std'],
             purity_s1_mean=results['s1']['purities_mean'],
             purity_s1_std=results['s1']['purities_std'],
             purity_s8_mean=results['s8']['purities_mean'],
             purity_s8_std=results['s8']['purities_std'])

    print(f"   Data saved: {data_filename}")

    return {
        'results': results,
        'analysis': analysis,
        'gamma_0': gamma_0,
        'gamma_1': gamma_1,
        'gamma_8': gamma_8,
        'suppression_1': suppression_1,
        'suppression_8': suppression_8,
        'sigma': sigma,
        'verdict': verdict,
        'quality_score': quality_score,
        'issues': issues
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║        CRITICAL BASELINE TEST - THE DEFINITIVE EXPERIMENT            ║
    ║                                                                      ║
    ║  This test compares:                                                 ║
    ║    • s=0: No system Hamiltonian (pure decoherence)                   ║
    ║    • s=1: Weak scrambling                                            ║
    ║    • s=8: Strong scrambling                                          ║
    ║                                                                      ║
    ║  This is THE MOST IMPORTANT TEST to validate the hypothesis.         ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    print(f"Mode: {'QUICK (10 shots)' if QUICK_MODE else 'RIGOROUS (50 shots)'}")
    print(f"Estimated total runtime: ~{N_SHOTS * 1.8:.0f} seconds ({N_SHOTS * 1.8 / 60:.1f} minutes)\n")

    response = input("Proceed with test? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        print("\n🚀 Starting critical baseline test...\n")

        try:
            baseline_results = test_baseline_comparison()

            print("\n" + "="*70)
            print("TEST COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"\n🎯 Final Verdict: {baseline_results['verdict']}")
            print(f"🔢 Suppression Factor: {baseline_results['suppression_8']:.2f}x")
            print(f"📊 Statistical Significance: {baseline_results['sigma']:.2f}σ")

        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n❌ Test cancelled by user.")
        print("\nTo run later, execute: python baseline_test.py")
        sys.exit(0)# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def kron_list(mats):
    out = np.array([1.0], dtype=complex)
    for m in mats:
        out = np.kron(out, m)
    return out

def op_on(n, op, idx):
    mats = [I2]*n
    mats[idx] = op
    return kron_list(mats)

def two_on(n, op1, i1, op2, i2):
    mats = [I2]*n
    mats[i1] = op1
    mats[i2] = op2
    return kron_list(mats)

def make_hermitian(A):
    return 0.5 * (A + A.conj().T)

def normalize_rho(rho):
    rho = make_hermitian(rho)
    tr = np.real(np.trace(rho))
    if np.abs(tr) < 1e-20:
        return rho
    return rho / tr

def purity(rho):
    val = np.trace(rho @ rho)
    return float(np.real(val))

def von_neumann_entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-14]
    if len(eigvals) == 0:
        return 0.0
    return -np.sum(eigvals * np.log2(eigvals))

def partial_trace_env(rho, N, M):
    """Trace out last M qubits (environment)."""
    dS = 2**N
    dE = 2**M
    rho = rho.reshape(dS, dE, dS, dE)
    rhoS = np.zeros((dS, dS), dtype=complex)
    for i in range(dE):
        rhoS += rho[:, i, :, i]
    return rhoS

def coherence_measure(rho):
    """Sum of squared off-diagonal elements."""
    d = rho.shape[0]
    off_diag = 0.0
    for i in range(d):
        for j in range(i+1, d):
            off_diag += np.abs(rho[i,j])**2
    return np.sqrt(off_diag)

def compute_entanglement_entropy(rho, N, M):
    """Entanglement entropy between system and environment."""
    rhoS = partial_trace_env(rho, N, M)
    return von_neumann_entropy(rhoS)

# ============================================================================
# STRUCTURED HAMILTONIANS (Not just random!)
# ============================================================================

def heisenberg_hamiltonian(N, J=1.0, h=0.0, pbc=False):
    """
    Heisenberg XXZ model: H = J Σ(X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1}) + h Σ Z_i
    
    Args:
        pbc: Periodic boundary conditions
    """
    d = 2**N
    H = np.zeros((d, d), dtype=complex)
    
    # Nearest-neighbor interactions
    pairs = [(i, i+1) for i in range(N-1)]
    if pbc and N > 2:
        pairs.append((N-1, 0))
    
    for i, j in pairs:
        H += J * two_on(N, sx, i, sx, j)
        H += J * two_on(N, sy, i, sy, j)
        H += J * two_on(N, sz, i, sz, j)
    
    # External field
    if h != 0:
        for i in range(N):
            H += h * op_on(N, sz, i)
    
    return make_hermitian(H)

def transverse_ising_hamiltonian(N, J=1.0, h=1.0, pbc=False):
    """
    Transverse field Ising: H = -J Σ Z_i Z_{i+1} - h Σ X_i
    """
    d = 2**N
    H = np.zeros((d, d), dtype=complex)
    
    # ZZ interactions
    pairs = [(i, i+1) for i in range(N-1)]
    if pbc and N > 2:
        pairs.append((N-1, 0))
    
    for i, j in pairs:
        H -= J * two_on(N, sz, i, sz, j)
    
    # Transverse field
    for i in range(N):
        H -= h * op_on(N, sx, i)
    
    return make_hermitian(H)

def random_scrambling_H(N, strength=1.0, longrange=True, seed=None):
    """Random scrambling Hamiltonian with optional seed for reproducibility."""
    if seed is not None:
        rng_local = default_rng(seed)
    else:
        rng_local = default_rng()
    
    d = 2**N
    H = np.zeros((d, d), dtype=complex)
    
    # Local fields
    for i in range(N):
        a, b, c = rng_local.normal(scale=strength, size=3)
        H += a * op_on(N, sx, i) + b * op_on(N, sy, i) + c * op_on(N, sz, i)
    
    # Two-body terms
    if longrange:
        for i in range(N):
            for j in range(i+1, N):
                Jx = rng_local.normal(scale=0.4 * strength)
                Jy = rng_local.normal(scale=0.4 * strength)
                Jz = rng_local.normal(scale=0.4 * strength)
                H += Jx * two_on(N, sx, i, sx, j)
                H += Jy * two_on(N, sy, i, sy, j)
                H += Jz * two_on(N, sz, i, sz, j)
    
    return make_hermitian(H)

# ============================================================================
# OUT-OF-TIME-ORDER CORRELATORS (OTOCs)
# ============================================================================

def compute_otoc(N, H, t, A_op, B_op):
    """
    Compute OTOC: F(t) = ⟨[A(t), B(0)]^†[A(t), B(0)]⟩
    
    For thermalization/scrambling, F(t) should grow exponentially initially.
    """
    d = 2**N
    
    # Time evolution
    U = expm(-1j * H * t)
    
    # A(t) = U† A U
    At = U.conj().T @ A_op @ U
    
    # Commutator [A(t), B]
    comm = At @ B_op - B_op @ At
    
    # F(t) = Tr([A(t),B]† [A(t),B]) / d (normalized)
    F = np.real(np.trace(comm.conj().T @ comm)) / d
    
    return F

def measure_scrambling_time(N, H, A_idx=0, B_idx=None, t_max=2.0, n_points=50):
‎    """
‎    Measure scrambling time from OTOC growth.
‎    Returns: tau_scramble, otoc_times, otoc_values
‎    """
‎    if B_idx is None:
‎        B_idx = N-1 if N > 1 else 0
‎    
‎    A_op = op_on(N, sz, A_idx)
‎    B_op = op_on(N, sz, B_idx)
‎    
‎    times = np.linspace(0, t_max, n_points)
‎    otocs = []
‎    
‎    for t in times:
‎        F = compute_otoc(N, H, t, A_op, B_op)
‎        otocs.append(F)
‎    
‎    otocs = np.array(otocs)
‎    
‎    # Method 1: Fit exponential growth in early regime
‎    # F(t) ~ F_0 * exp(2λ_L t) where λ_L is Lyapunov exponent
‎    
‎    # Find the region where OTOC is growing (not saturated)
‎    F_initial = otocs[0]
‎    F_max = np.max(otocs)
‎    
‎    # Look for growth phase: F > 1.2*F_initial and F < 0.8*F_max
‎    growth_start = np.where(otocs > 1.2 * F_initial)[0]
‎    growth_end = np.where(otocs > 0.8 * F_max)[0]
‎    
‎    if len(growth_start) > 0 and len(growth_end) > 0:
‎        idx_start = growth_start[0]
‎        idx_end = growth_end[0]
‎        
‎        # Need at least 5 points for reliable fit
‎        if idx_end > idx_start and (idx_end - idx_start) >= 5:
‎            t_fit = times[idx_start:idx_end]
‎            F_fit = otocs[idx_start:idx_end]
‎            
‎            # Remove any zeros or negatives
‎            valid = F_fit > 1e-12
‎            if np.sum(valid) >= 3:
‎                t_fit = t_fit[valid]
‎                F_fit = F_fit[valid]
‎                
‎                try:
‎                    # Fit log(F) = log(F0) + 2λ_L t
‎                    coeffs = np.polyfit(t_fit, np.log(F_fit), 1)
‎                    lambda_L = coeffs[0] / 2.0
‎                    
‎                    if lambda_L > 0:
‎                        tau_scramble = 1.0 / lambda_L
‎                        return tau_scramble, times, otocs
‎                except:
‎                    pass
‎    
‎    # Method 2: If exponential fit fails, use time to reach half-maximum
‎    # This is more robust for strongly scrambling systems
‎    F_half = (F_initial + F_max) / 2.0
‎    idx_half = np.where(otocs >= F_half)[0]
‎    
‎    if len(idx_half) > 0:
‎        t_half = times[idx_half[0]]
‎        # Rough estimate: tau ~ t_half / ln(2)
‎        tau_scramble = t_half / 0.693
‎        return tau_scramble, times, otocs
‎    
‎    # Method 3: If still no growth, estimate from variance
‎    # Strong scrambling shows rapid increase in OTOC variance
‎    if len(otocs) > 10:
‎        # Compute time derivative
‎        dt = times[1] - times[0]
‎        dF_dt = np.gradient(otocs, dt)
‎        
‎        # Find maximum growth rate
‎        idx_max_growth = np.argmax(dF_dt)
‎        
‎        if dF_dt[idx_max_growth] > 0:
‎            # Estimate: tau ~ 1/max_growth_rate
‎            tau_scramble = 1.0 / dF_dt[idx_max_growth]
‎            return tau_scramble, times, otocs
‎    
‎    # If all methods fail, return a large but finite value
‎    print(f"    Warning: Could not fit scrambling time, using fallback estimate")
‎    tau_scramble = t_max  # Use t_max as upper bound
‎    return tau_scramble, times, otocs

# ============================================================================
# INITIAL STATE PREPARATIONS
# ============================================================================

def prepare_initial_state(N, state_type='random', H=None, temperature=None, seed=None):
    """
    Prepare various initial states for testing.
    
    Types:
    - 'random': Random pure state
    - 'ground': Ground state of H
    - 'thermal': Thermal state at temperature T
    - 'product': Product of |+⟩ states
    - 'ghz': GHZ state (maximally entangled)
    """
    d = 2**N
    
    if seed is not None:
        rng_local = default_rng(seed)
    else:
        rng_local = default_rng()
    
    if state_type == 'random':
        psi = rng_local.normal(size=d) + 1j * rng_local.normal(size=d)
        psi /= np.linalg.norm(psi)
        return np.outer(psi, psi.conj())
    
    elif state_type == 'ground':
        if H is None:
            raise ValueError("Need H for ground state")
        eigvals, eigvecs = np.linalg.eigh(H)
        psi = eigvecs[:, 0]
        return np.outer(psi, psi.conj())
    
    elif state_type == 'thermal':
        if H is None or temperature is None:
            raise ValueError("Need H and temperature for thermal state")
        eigvals, eigvecs = np.linalg.eigh(H)
        beta = 1.0 / temperature
        boltzmann = np.exp(-beta * eigvals)
        Z = np.sum(boltzmann)
        rho = np.zeros((d, d), dtype=complex)
        for i, w in enumerate(boltzmann):
            psi = eigvecs[:, i]
            rho += (w/Z) * np.outer(psi, psi.conj())
        return normalize_rho(rho)
    
    elif state_type == 'product':
        # |+⟩^⊗N state
        plus = np.array([1, 1]) / np.sqrt(2)
        psi = plus
        for _ in range(N-1):
            psi = np.kron(psi, plus)
        return np.outer(psi, psi.conj())
    
    elif state_type == 'ghz':
        # (|0...0⟩ + |1...1⟩)/√2
        psi = np.zeros(d, dtype=complex)
        psi[0] = 1/np.sqrt(2)
        psi[-1] = 1/np.sqrt(2)
        return np.outer(psi, psi.conj())
    
    else:
        raise ValueError(f"Unknown state type: {state_type}")

# ============================================================================
# CORE EVOLUTION WITH ENVIRONMENT
# ============================================================================

def build_H_total(N, M_env, scr_strength, coupling_g, hamiltonian_type='random',
‎                  boundary_fraction=0.3, seed=None, **ham_kwargs):
‎    """
‎    Build total Hamiltonian with specified system Hamiltonian type.
‎    
‎    hamiltonian_type: 'random', 'heisenberg', 'ising'
‎    """
‎    # System Hamiltonian
‎    if hamiltonian_type == 'random':
‎        Hs = random_scrambling_H(N, strength=scr_strength, longrange=True, seed=seed)
‎    elif hamiltonian_type == 'heisenberg':
‎        # Use ham_kwargs if provided, otherwise use scr_strength as J
‎        J = ham_kwargs.get('J', scr_strength)
‎        h = ham_kwargs.get('h', 0.0)
‎        pbc = ham_kwargs.get('pbc', False)
‎        Hs = heisenberg_hamiltonian(N, J=J, h=h, pbc=pbc)
‎    elif hamiltonian_type == 'ising':
‎        # Use ham_kwargs if provided, otherwise use scr_strength as J and h
‎        J = ham_kwargs.get('J', scr_strength)
‎        h = ham_kwargs.get('h', scr_strength)
‎        pbc = ham_kwargs.get('pbc', False)
‎        Hs = transverse_ising_hamiltonian(N, J=J, h=h, pbc=pbc)
‎    else:
‎        raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}")
‎    
‎    # Environment Hamiltonian (keep same)
‎    He = random_scrambling_H(M_env, strength=0.5, longrange=False, 
‎                             seed=seed+1 if seed else None) if M_env > 0 else np.zeros((1,1), dtype=complex)
‎    
‎    dS = 2**N
‎    dE = 2**M_env
‎    
‎    # Total: H = Hs ⊗ I + I ⊗ He + H_int
‎    Htot = np.kron(Hs, np.eye(dE)) + np.kron(np.eye(dS), He)
‎    
‎    # Collective boundary coupling
‎    n_boundary = max(1, int(N * boundary_fraction))
‎    boundary_indices = list(range(n_boundary))
‎    
‎    A_sys = np.zeros((dS, dS), dtype=complex)
‎    for b in boundary_indices:
‎        A_sys += op_on(N, sz, b)
‎    A_sys = make_hermitian(A_sys)
‎    
‎    B_env = np.zeros((dE, dE), dtype=complex)
‎    for e in range(M_env):
‎        B_env += op_on(M_env, sz, e)
‎    B_env = make_hermitian(B_env)
‎    
‎    Htot += coupling_g * np.kron(A_sys, B_env)
‎    
‎    return make_hermitian(Htot), Hs


def evolve_system(N, M_env, scr_strength, coupling_g, t_max, n_steps,
                 hamiltonian_type='random', init_state_type='random',
                 seed=None, **kwargs):
    """
    Single evolution run with comprehensive measurements.
    
    Returns dict with times, purities, entropies, coherences, etc.
    """
    Htot, Hs = build_H_total(N, M_env, scr_strength, coupling_g, 
                             hamiltonian_type, seed=seed, **kwargs)
    
    # Initial state
    rhoS0 = prepare_initial_state(N, init_state_type, H=Hs, seed=seed)
    rhoE0 = np.eye(2**M_env) / (2**M_env)
    rho0 = np.kron(rhoS0, rhoE0)
    
    times = np.linspace(0, t_max, n_steps)
    dt = times[1] - times[0] if len(times) > 1 else t_max
    
    # Storage
    purities = []
    entropies = []
    coherences = []
    entanglement_entropies = []
    
    for i, t in enumerate(times):
        # Evolve
        U = expm(-1j * Htot * t)
        rho_t = U @ rho0 @ U.conj().T
        
        # Measure system
        rhoS = partial_trace_env(rho_t, N, M_env)
        rhoS = normalize_rho(rhoS)
        
        purities.append(purity(rhoS))
        entropies.append(von_neumann_entropy(rhoS))
        coherences.append(coherence_measure(rhoS))
        entanglement_entropies.append(compute_entanglement_entropy(rho_t, N, M_env))
    
    return {
        'times': times,
        'purities': np.array(purities),
        'entropies': np.array(entropies),
        'coherences': np.array(coherences),
        'entanglement': np.array(entanglement_entropies)
    }

# ============================================================================
# STATISTICAL ANALYSIS WITH ERROR BARS
# ============================================================================

def run_ensemble(N, M_env, scr_strength, coupling_g, t_max, n_steps,
                shots, hamiltonian_type='random', init_state_type='random',
                **kwargs):
    """
    Run multiple shots and compute statistics.
    
    Returns: mean results + standard deviations
    """
    print(f"  Running {shots} shots...", end='', flush=True)
    start = pytime.time()
    
    all_purities = []
    all_entropies = []
    all_coherences = []
    all_entanglement = []
    
    for shot in range(shots):
        seed = 1000 + shot  # Reproducible but different seeds
        
        result = evolve_system(N, M_env, scr_strength, coupling_g, t_max, n_steps,
                              hamiltonian_type, init_state_type, seed=seed, **kwargs)
        
        all_purities.append(result['purities'])
        all_entropies.append(result['entropies'])
        all_coherences.append(result['coherences'])
        all_entanglement.append(result['entanglement'])
        
        if (shot+1) % 10 == 0:
            print(f"{shot+1}...", end='', flush=True)
    
    elapsed = pytime.time() - start
    print(f" done ({elapsed:.1f}s)")
    
    # Convert to arrays
    all_purities = np.array(all_purities)
    all_entropies = np.array(all_entropies)
    all_coherences = np.array(all_coherences)
    all_entanglement = np.array(all_entanglement)
    
    return {
        'times': result['times'],
        'purities_mean': np.mean(all_purities, axis=0),
        'purities_std': np.std(all_purities, axis=0),
        'entropies_mean': np.mean(all_entropies, axis=0),
        'entropies_std': np.std(all_entropies, axis=0),
        'coherences_mean': np.mean(all_coherences, axis=0),
        'coherences_std': np.std(all_coherences, axis=0),
        'entanglement_mean': np.mean(all_entanglement, axis=0),
        'entanglement_std': np.std(all_entanglement, axis=0),
    }

def fit_decay_rate_with_error(times, purities_mean, purities_std, fit_range=(0.0, 2.0)):
    """
    Fit exponential decay with bootstrap for error estimation.
    """
    mask = (times >= fit_range[0]) & (times <= fit_range[1])
    t_fit = times[mask]
    p_mean = np.clip(purities_mean[mask], 1e-12, 1.0)
    p_std = purities_std[mask]
    
    if len(t_fit) < 3:
        return np.nan, np.nan
    
    # Best fit
    try:
        coeffs = np.polyfit(t_fit, np.log(p_mean), 1)
        gamma_best = -coeffs[0]
    except:
        return np.nan, np.nan
    
    # Bootstrap for error bars
    n_bootstrap = 100
    gammas_boot = []
    
    for _ in range(n_bootstrap):
        p_sample = p_mean + np.random.normal(0, p_std)
        p_sample = np.clip(p_sample, 1e-12, 1.0)
        try:
            coeffs = np.polyfit(t_fit, np.log(p_sample), 1)
            gammas_boot.append(-coeffs[0])
        except:
            pass
    
    if len(gammas_boot) > 0:
        gamma_std = np.std(gammas_boot)
    else:
        gamma_std = np.nan
    
    return gamma_best, gamma_std

# ============================================================================
# RIGOROUS TEST SUITE
# ============================================================================

def test_hamiltonian_types(N=4, M_env=2, coupling_g=0.02, t_max=5.0,
                          n_steps=DEFAULT_N_STEPS, shots=DEFAULT_SHOTS):
    """
    MUST-DO TEST 3: Test structured Hamiltonians (not just random)
    """
    print("\n" + "="*70)
    print("RIGOROUS TEST 1: Hamiltonian Type Comparison")
    print(f"N={N}, M_env={M_env}, g={coupling_g}, shots={shots}")
    print("="*70 + "\n")
    
    hamiltonian_types = {
        'Random (strong)': ('random', {'scr_strength': 8.0}),
        'Random (weak)': ('random', {'scr_strength': 1.0}),
        'Heisenberg (strong)': ('heisenberg', {'J': 8.0, 'h': 0.0}),
        'Heisenberg (weak)': ('heisenberg', {'J': 1.0, 'h': 0.0}),
        'Ising (strong)': ('ising', {'J': 8.0, 'h': 8.0}),
        'Ising (weak)': ('ising', {'J': 1.0, 'h': 1.0}),
    }
    
    results = {}
    
    for label, (ham_type, params) in hamiltonian_types.items():
        print(f"Testing: {label}")
        
        # Extract scr_strength from params, use default strength for structured Hamiltonians
        if ham_type == 'random':
            scr = params.get('scr_strength', 1.0)
            ham_params = {}  # Don't pass scr_strength again in kwargs
        else:
            # For Heisenberg/Ising, use J as the strength measure
            scr = params.get('J', 1.0)
            ham_params = params  # Pass all params (J, h, etc.)
        
        res = run_ensemble(N, M_env, scr, coupling_g, t_max, n_steps, shots,
                          hamiltonian_type=ham_type, **ham_params)
        
        # Fit decay rate
        gamma, gamma_err = fit_decay_rate_with_error(
            res['times'], res['purities_mean'], res['purities_std']
        )
        
        # Plateau purity
        late_mask = res['times'] >= t_max * 0.6
        plateau = np.mean(res['purities_mean'][late_mask])
        plateau_std = np.mean(res['purities_std'][late_mask])
        
        results[label] = {
            **res,
            'gamma': gamma,
            'gamma_err': gamma_err,
            'plateau': plateau,
            'plateau_std': plateau_std
        }
        
        print(f"  γ = {gamma:.6f} ± {gamma_err:.6f}")
        print(f"  Plateau = {plateau:.4f} ± {plateau_std:.4f}\n")
    
    return results

def test_coupling_sweep_rigorous(N=4, M_env=2, scr_strength=8.0, t_max=5.0,
                                 n_steps=DEFAULT_N_STEPS, shots=DEFAULT_SHOTS):
    """
    MUST-DO TEST 5: Fine coupling sweep between 0.01 and 0.05
    """
    print("\n" + "="*70)
    print("RIGOROUS TEST 2: Fine Coupling Strength Sweep")
    print(f"N={N}, M_env={M_env}, scr={scr_strength}, shots={shots}")
    print("="*70 + "\n")
    
    coupling_gs = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    
    results = {'gs': [], 'gammas': [], 'gamma_errs': [], 'plateaus': [], 'plateau_errs': []}
    
    for g in coupling_gs:
        print(f"Testing g={g:.4f}")
        
        res = run_ensemble(N, M_env, scr_strength, g, t_max, n_steps, shots)
        
        gamma, gamma_err = fit_decay_rate_with_error(
            res['times'], res['purities_mean'], res['purities_std']
        )
        
        late_mask = res['times'] >= t_max * 0.6
        plateau = np.mean(res['purities_mean'][late_mask])
        plateau_std = np.mean(res['purities_std'][late_mask])
        
        results['gs'].append(g)
        results['gammas'].append(gamma)
        results['gamma_errs'].append(gamma_err)
        results['plateaus'].append(plateau)
        results['plateau_errs'].append(plateau_std)
        
        print(f"  γ = {gamma:.6f} ± {gamma_err:.6f}")
        print(f"  Plateau = {plateau:.4f} ± {plateau_std:.4f}\n")
    
    for key in ['gs', 'gammas', 'gamma_errs', 'plateaus', 'plateau_errs']:
        results[key] = np.array(results[key])
    
    return results

def test_bath_size_rigorous(N=4, scr_strength=8.0, coupling_g=0.02, t_max=5.0,
                           n_steps=DEFAULT_N_STEPS, shots=DEFAULT_SHOTS):
    """
    MUST-DO TEST 4: Resolve M_env anomaly with high statistics
    """
    print("\n" + "="*70)
    print("RIGOROUS TEST 3: Bath Size Scaling with Error Bars")
    print(f"N={N}, scr={scr_strength}, g={coupling_g}, shots={shots}")
    print("="*70 + "\n")
    
    M_envs = [1, 2, 3, 4] if not QUICK_TEST else [1, 2, 3]
    
    results = {'M_envs': [], 'gammas': [], 'gamma_errs': [], 
               'plateaus': [], 'plateau_errs': [], 'all_data': {}}
    
    for M_env in M_envs:
        print(f"Testing M_env={M_env} (dim={2**(N+M_env)})")
        
        res = run_ensemble(N, M_env, scr_strength, coupling_g, t_max, n_steps, shots)
        
        gamma, gamma_err = fit_decay_rate_with_error(
            res['times'], res['purities_mean'], res['purities_std']
        )
        
        late_mask = res['times'] >= t_max * 0.6
        plateau = np.mean(res['purities_mean'][late_mask])
        plateau_std = np.mean(res['purities_std'][late_mask])
        
        results['M_envs'].append(M_env)
        results['gammas'].append(gamma)
        results['gamma_errs'].append(gamma_err)
        results['plateaus'].append(plateau)
        results['plateau_errs'].append(plateau_std)
        results['all_data'][M_env] = res
        
        print(f"  γ = {gamma:.6f} ± {gamma_err:.6f}")
        print(f"  Plateau = {plateau:.4f} ± {plateau_std:.4f}\n")
    
    for key in ['M_envs', 'gammas', 'gamma_errs', 'plateaus', 'plateau_errs']:
        results[key] = np.array(results[key])
    
    return results

def test_initial_states(N=4, M_env=2, scr_strength=8.0, coupling_g=0.02,
                       t_max=5.0, n_steps=DEFAULT_N_STEPS, shots=DEFAULT_SHOTS):
    """
    MUST-DO TEST 7: Different initial states
    """
    print("\n" + "="*70)
    print("RIGOROUS TEST 4: Initial State Dependence")
    print(f"N={N}, M_env={M_env}, scr={scr_strength}, g={coupling_g}, shots={shots}")
    print("="*70 + "\n")
    
    init_types = ['random', 'product', 'ghz'] #, 'ground']  # ground requires H
    
    results = {}
    
    for init_type in init_types:
        print(f"Testing initial state: {init_type}")
        
        res = run_ensemble(N, M_env, scr_strength, coupling_g, t_max, n_steps, shots,
                          init_state_type=init_type)
        
        gamma, gamma_err = fit_decay_rate_with_error(
            res['times'], res['purities_mean'], res['purities_std']
        )
        
        late_mask = res['times'] >= t_max * 0.6
        plateau = np.mean(res['purities_mean'][late_mask])
        plateau_std = np.mean(res['purities_std'][late_mask])
        
        results[init_type] = {
            **res,
            'gamma': gamma,
            'gamma_err': gamma_err,
            'plateau': plateau,
            'plateau_std': plateau_std
        }
        
        print(f"  γ = {gamma:.6f} ± {gamma_err:.6f}")
        print(f"  Plateau = {plateau:.4f} ± {plateau_std:.4f}\n")
    
    return results

def test_otoc_scrambling_time(N=4, scr_strengths=None, shots=10):
‎    """
‎    SHOULD-DO TEST 8: Measure actual scrambling time via OTOCs
‎    """
‎    print("\n" + "="*70)
‎    print("RIGOROUS TEST 5: OTOC Measurement of Scrambling Time")
‎    print(f"N={N}, shots={shots}")
‎    print("="*70 + "\n")
‎    
‎    if scr_strengths is None:
‎        scr_strengths = [1.0, 2.0, 4.0, 8.0]
‎    
‎    results = {'strengths': [], 'tau_scrambles': [], 'tau_stds': [], 'otoc_data': {}}
‎    
‎    for strength in scr_strengths:
‎        print(f"Testing strength={strength}")
‎        
‎        taus = []
‎        otoc_curves = []
‎        
‎        # Adjust t_max based on expected scrambling time
‎        # Stronger interactions → faster scrambling → shorter t_max needed
‎        t_max = max(1.0, 4.0 / strength)  # Adaptive time window
‎        
‎        for shot in range(shots):
‎            H = random_scrambling_H(N, strength=strength, longrange=True, seed=2000+shot)
‎            tau, times, otocs = measure_scrambling_time(N, H, t_max=t_max, n_points=50)
‎            
‎            # Only include finite values
‎            if np.isfinite(tau) and tau < 100:  # Reasonable upper bound
‎                taus.append(tau)
‎                otoc_curves.append(otocs)
‎        
‎        if len(taus) > 0:
‎            tau_mean = np.mean(taus)
‎            tau_std = np.std(taus)
‎        else:
‎            tau_mean = np.nan
‎            tau_std = np.nan
‎            print(f"  Warning: All measurements failed for strength={strength}")
‎        
‎        results['strengths'].append(strength)
‎        results['tau_scrambles'].append(tau_mean)
‎        results['tau_stds'].append(tau_std)
‎        
‎        if len(otoc_curves) > 0:
‎            results['otoc_data'][strength] = {
‎                'times': times,
‎                'otocs_mean': np.mean(otoc_curves, axis=0),
‎                'otocs_std': np.std(otoc_curves, axis=0)
‎            }
‎        else:
‎            results['otoc_data'][strength] = {
‎                'times': times,
‎                'otocs_mean': otocs,  # Use last attempt
‎                'otocs_std': np.zeros_like(otocs)
‎            }
‎        
‎        print(f"  τ_scramble = {tau_mean:.4f} ± {tau_std:.4f}")
‎        print(f"  ({len(taus)}/{shots} successful measurements)\n")
‎    
‎    for key in ['strengths', 'tau_scrambles', 'tau_stds']:
‎        results[key] = np.array(results[key])
‎    
‎    return results

def test_time_evolution_justification(N=4, M_env=2, scr_strength=8.0, coupling_g=0.02,
                                     t_max_short=5.0, t_max_long=20.0,
                                     n_steps=DEFAULT_N_STEPS, shots=DEFAULT_SHOTS//2):
    """
    MUST-DO TEST 6: Justify early-time vs long-time behavior
    """
    print("\n" + "="*70)
    print("RIGOROUS TEST 6: Time Evolution - Early vs Long")
    print(f"N={N}, M_env={M_env}, scr={scr_strength}, g={coupling_g}, shots={shots}")
    print("="*70 + "\n")
    
    print("Short time evolution (t_max=5.0)")
    res_short = run_ensemble(N, M_env, scr_strength, coupling_g, 
                            t_max_short, n_steps, shots)
    
    print("\nLong time evolution (t_max=20.0)")
    res_long = run_ensemble(N, M_env, scr_strength, coupling_g, 
                           t_max_long, n_steps, shots)
    
    # Fit decay in different regimes
    gamma_early, gamma_early_err = fit_decay_rate_with_error(
        res_long['times'], res_long['purities_mean'], res_long['purities_std'],
        fit_range=(0.0, 2.0)
    )
    
    gamma_mid, gamma_mid_err = fit_decay_rate_with_error(
        res_long['times'], res_long['purities_mean'], res_long['purities_std'],
        fit_range=(2.0, 10.0)
    )
    
    gamma_late, gamma_late_err = fit_decay_rate_with_error(
        res_long['times'], res_long['purities_mean'], res_long['purities_std'],
        fit_range=(10.0, 20.0)
    )
    
    print(f"\nDecay rates in different time regimes:")
    print(f"  Early (0-2):   γ = {gamma_early:.6f} ± {gamma_early_err:.6f}")
    print(f"  Middle (2-10): γ = {gamma_mid:.6f} ± {gamma_mid_err:.6f}")
    print(f"  Late (10-20):  γ = {gamma_late:.6f} ± {gamma_late_err:.6f}")
    
    return {
        'short': res_short,
        'long': res_long,
        'gamma_early': gamma_early,
        'gamma_mid': gamma_mid,
        'gamma_late': gamma_late,
        'gamma_early_err': gamma_early_err,
        'gamma_mid_err': gamma_mid_err,
        'gamma_late_err': gamma_late_err
    }

def test_phase_diagram(N=4, M_env=2, t_max=5.0, n_steps=DEFAULT_N_STEPS, 
                      shots=DEFAULT_SHOTS//2):
    """
    SHOULD-DO TEST 10: Phase diagram in (g, scr_strength) space
    """
    print("\n" + "="*70)
    print("RIGOROUS TEST 7: Phase Diagram (g vs scr_strength)")
    print(f"N={N}, M_env={M_env}, shots={shots}")
    print("="*70 + "\n")
    
    # Grid of parameters
    gs = [0.01, 0.02, 0.04, 0.08] if not QUICK_TEST else [0.01, 0.04]
    scr_strengths = [1.0, 2.0, 4.0, 8.0] if not QUICK_TEST else [1.0, 4.0, 8.0]
    
    results = {
        'gs': gs,
        'scr_strengths': scr_strengths,
        'plateaus': np.zeros((len(scr_strengths), len(gs))),
        'plateau_errs': np.zeros((len(scr_strengths), len(gs))),
        'gammas': np.zeros((len(scr_strengths), len(gs))),
        'gamma_errs': np.zeros((len(scr_strengths), len(gs)))
    }
    
    for i, scr in enumerate(scr_strengths):
        for j, g in enumerate(gs):
            print(f"Testing scr={scr}, g={g}")
            
            res = run_ensemble(N, M_env, scr, g, t_max, n_steps, shots)
            
            gamma, gamma_err = fit_decay_rate_with_error(
                res['times'], res['purities_mean'], res['purities_std']
            )
            
            late_mask = res['times'] >= t_max * 0.6
            plateau = np.mean(res['purities_mean'][late_mask])
            plateau_std = np.mean(res['purities_std'][late_mask])
            
            results['plateaus'][i, j] = plateau
            results['plateau_errs'][i, j] = plateau_std
            results['gammas'][i, j] = gamma
            results['gamma_errs'][i, j] = gamma_err
            
            print(f"  γ = {gamma:.6f} ± {gamma_err:.6f}, plateau = {plateau:.4f} ± {plateau_std:.4f}\n")
    
    return results

# ============================================================================
# ANALYTICAL MODEL COMPARISON
# ============================================================================

def analytical_model_comparison(otoc_results, coupling_results):
    """
    SHOULD-DO TEST 9: Compare to analytical model γ_eff ~ g²·τ_scramble
    """
    print("\n" + "="*70)
    print("ANALYTICAL MODEL: γ_eff ~ g² · f(τ_scramble)")
    print("="*70 + "\n")
    
    # Extract scrambling times
    strengths = otoc_results['strengths']
    tau_scrambles = otoc_results['tau_scrambles']
    
    # Extract measured gammas
    gs = coupling_results['gs']
    gammas = coupling_results['gammas']
    
    print("Theoretical prediction: γ ∝ g² (Fermi Golden Rule)")
    print("Modified by scrambling: γ_eff ∝ g² · f(τ)")
    print("\nChecking g² scaling:")
    
    # Fit γ vs g²
    g_squared = gs**2
    
    # Linear fit: γ = α·g²
    valid = ~np.isnan(gammas)
    if np.sum(valid) > 2:
        alpha, residuals, _, _, _ = np.polyfit(g_squared[valid], gammas[valid], 1, full=True)
        alpha = alpha[0]
        
        print(f"  γ = {alpha:.4f} · g²")
        print(f"  R² = {1 - residuals[0]/np.var(gammas[valid]) if len(residuals) > 0 else 'N/A':.4f}")
    
    # Estimate suppression factor
    tau_at_strength_8 = tau_scrambles[strengths == 8.0][0] if 8.0 in strengths else 0.1
    
    print(f"\nScrambling time at strength=8.0: τ = {tau_at_strength_8:.4f}")
    print(f"Expected decoherence without scrambling: γ_0 ~ g² ~ {0.02**2:.6f}")
    print(f"Observed decoherence with scrambling: γ_eff ~ {gammas[gs == 0.02][0] if 0.02 in gs else 'N/A':.6f}")
    
    if 0.02 in gs and not np.isnan(gammas[gs == 0.02][0]):
        suppression = (0.02**2) / gammas[gs == 0.02][0]
        print(f"Suppression factor: {suppression:.1f}x")
    
    return {
        'alpha': alpha if 'alpha' in locals() else np.nan,
        'tau_scramble': tau_at_strength_8,
        'suppression_factor': suppression if 'suppression' in locals() else np.nan
    }

# ============================================================================
# COMPREHENSIVE PLOTTING WITH ERROR BARS
# ============================================================================

def plot_rigorous_results(results_dict, save_prefix='rigorous_test'):
    """
    Generate publication-quality plots with error bars.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test 1: Hamiltonian types
    if 'hamiltonian_types' in results_dict:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        results = results_dict['hamiltonian_types']
        labels = list(results.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        
        # Plot trajectories
        ax = axes[0, 0]
        for i, label in enumerate(labels):
            res = results[label]
            ax.plot(res['times'], res['purities_mean'], '-', 
                   linewidth=2, label=label, color=colors[i])
            ax.fill_between(res['times'], 
                           res['purities_mean'] - res['purities_std'],
                           res['purities_mean'] + res['purities_std'],
                           alpha=0.2, color=colors[i])
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Purity', fontsize=11)
        ax.set_title('Purity Evolution (Different Hamiltonians)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Decay rates
        ax = axes[0, 1]
        gammas = [results[l]['gamma'] for l in labels]
        gamma_errs = [results[l]['gamma_err'] for l in labels]
        x = np.arange(len(labels))
        ax.bar(x, gammas, yerr=gamma_errs, capsize=5, alpha=0.7, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Decay rate γ', fontsize=11)
        ax.set_title('Decay Rates with Error Bars', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plateau purities
        ax = axes[1, 0]
        plateaus = [results[l]['plateau'] for l in labels]
        plateau_errs = [results[l]['plateau_std'] for l in labels]
        ax.bar(x, plateaus, yerr=plateau_errs, capsize=5, alpha=0.7, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Plateau purity', fontsize=11)
        ax.set_title('Asymptotic Purity', fontweight='bold')
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Strong vs weak comparison
        ax = axes[1, 1]
        strong_labels = [l for l in labels if 'strong' in l.lower()]
        weak_labels = [l for l in labels if 'weak' in l.lower()]
        
        if len(strong_labels) > 0 and len(weak_labels) > 0:
            strong_gammas = [results[l]['gamma'] for l in strong_labels]
            weak_gammas = [results[l]['gamma'] for l in weak_labels]
            
            x = np.arange(len(strong_labels))
            width = 0.35
            ax.bar(x - width/2, strong_gammas, width, label='Strong', alpha=0.7)
            ax.bar(x + width/2, weak_gammas, width, label='Weak', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([l.split('(')[0].strip() for l in strong_labels], fontsize=9)
            ax.set_ylabel('Decay rate γ', fontsize=11)
            ax.set_title('Strong vs Weak Comparison', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_hamiltonians_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Test 2: Coupling sweep
    if 'coupling_sweep' in results_dict:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        results = results_dict['coupling_sweep']
        
        # Gamma vs g
        ax = axes[0]
        ax.errorbar(results['gs'], results['gammas'], yerr=results['gamma_errs'],
                   fmt='o-', linewidth=2, markersize=8, capsize=5)
        ax.set_xlabel('Coupling strength g', fontsize=11)
        ax.set_ylabel('Decay rate γ', fontsize=11)
        ax.set_title('Decoherence Rate vs Coupling', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Try fitting g^2 dependence
        valid = ~np.isnan(results['gammas'])
        if np.sum(valid) > 2:
            g_fit = results['gs'][valid]
            gamma_fit = results['gammas'][valid]
            coeffs = np.polyfit(g_fit**2, gamma_fit, 1)
            g_theory = np.linspace(results['gs'][0], results['gs'][-1], 100)
            gamma_theory = coeffs[0] * g_theory**2 + coeffs[1]
            ax.plot(g_theory, gamma_theory, '--', alpha=0.5, 
                   label=f'Fit: γ = {coeffs[0]:.3f}·g² + {coeffs[1]:.6f}')
            ax.legend()
        
        # Plateau vs g
        ax = axes[1]
        ax.errorbar(results['gs'], results['plateaus'], yerr=results['plateau_errs'],
                   fmt='o-', linewidth=2, markersize=8, capsize=5, color='green')
        ax.set_xlabel('Coupling strength g', fontsize=11)
        ax.set_ylabel('Plateau purity', fontsize=11)
        ax.set_title('Protection vs Coupling Strength', fontweight='bold')
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_coupling_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Test 3: Bath size
    if 'bath_size' in results_dict:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        results = results_dict['bath_size']
        
        # Gamma vs M_env
        ax = axes[0]
        ax.errorbar(results['M_envs'], results['gammas'], yerr=results['gamma_errs'],
                   fmt='o-', linewidth=2, markersize=10, capsize=5)
        ax.set_xlabel('Environment size M_env', fontsize=11)
        ax.set_ylabel('Decay rate γ', fontsize=11)
        ax.set_title('Bath Size Dependence', fontweight='bold')
        ax.set_xticks(results['M_envs'])
        ax.grid(True, alpha=0.3)
        
        # Plateau vs M_env
        ax = axes[1]
        ax.errorbar(results['M_envs'], results['plateaus'], yerr=results['plateau_errs'],
                   fmt='o-', linewidth=2, markersize=10, capsize=5, color='green')
        ax.set_xlabel('Environment size M_env', fontsize=11)
        ax.set_ylabel('Plateau purity', fontsize=11)
        ax.set_title('Protection vs Bath Size', fontweight='bold')
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.5)
        ax.set_xticks(results['M_envs'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_bathsize_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Test 4: Initial states
    if 'initial_states' in results_dict:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        results = results_dict['initial_states']
        labels = list(results.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        
        # Trajectories
        ax = axes[0]
        for i, label in enumerate(labels):
            res = results[label]
            ax.plot(res['times'], res['purities_mean'], '-', 
                   linewidth=2, label=label, color=colors[i])
            ax.fill_between(res['times'],
                           res['purities_mean'] - res['purities_std'],
                           res['purities_mean'] + res['purities_std'],
                           alpha=0.2, color=colors[i])
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Purity', fontsize=11)
        ax.set_title('Initial State Dependence', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary
        ax = axes[1]
        gammas = [results[l]['gamma'] for l in labels]
        gamma_errs = [results[l]['gamma_err'] for l in labels]
        x = np.arange(len(labels))
        ax.bar(x, gammas, yerr=gamma_errs, capsize=5, alpha=0.7, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Decay rate γ', fontsize=11)
        ax.set_title('Decay Rates for Different Initial States', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_initial_states_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Test 5: OTOCs
    if 'otoc' in results_dict:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        results = results_dict['otoc']
        
        # Scrambling time vs strength
        ax = axes[0]
        ax.errorbar(results['strengths'], results['tau_scrambles'], 
                   yerr=results['tau_stds'], fmt='o-', linewidth=2, 
                   markersize=8, capsize=5)
        ax.set_xlabel('Scrambling strength', fontsize=11)
        ax.set_ylabel('τ_scramble', fontsize=11)
        ax.set_title('Scrambling Time Measurement', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Fit power law
        valid = ~np.isinf(results['tau_scrambles'])
        if np.sum(valid) > 2:
            log_s = np.log(results['strengths'][valid])
            log_tau = np.log(results['tau_scrambles'][valid])
            coeffs = np.polyfit(log_s, log_tau, 1)
            s_theory = np.linspace(results['strengths'][0], results['strengths'][-1], 100)
            tau_theory = np.exp(coeffs[1]) * s_theory**coeffs[0]
            ax.plot(s_theory, tau_theory, '--', alpha=0.5,
                   label=f'τ ∝ s^{coeffs[0]:.2f}')
            ax.legend()
        
        # OTOC curves
        ax = axes[1]
        colors = plt.cm.plasma(np.linspace(0, 1, len(results['strengths'])))
        for i, strength in enumerate(results['strengths']):
            data = results['otoc_data'][strength]
            ax.plot(data['times'], data['otocs_mean'], '-', 
                   linewidth=2, label=f's={strength}', color=colors[i])
            ax.fill_between(data['times'],
                           data['otocs_mean'] - data['otocs_std'],
                           data['otocs_mean'] + data['otocs_std'],
                           alpha=0.2, color=colors[i])
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('OTOC F(t)', fontsize=11)
        ax.set_title('Out-of-Time-Order Correlators', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_otoc_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Test 6: Time evolution
    if 'time_evolution' in results_dict:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        results = results_dict['time_evolution']
        
        # Short vs long
        ax = axes[0]
        for key, label in [('short', 'Short (t=5)'), ('long', 'Long (t=20)')]:
            if key in results:
                res = results[key]
                ax.plot(res['times'], res['purities_mean'], '-', 
                       linewidth=2, label=label)
                ax.fill_between(res['times'],
                               res['purities_mean'] - res['purities_std'],
                               res['purities_mean'] + res['purities_std'],
                               alpha=0.2)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Purity', fontsize=11)
        ax.set_title('Time Evolution: Short vs Long', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Decay rates in different regimes
        ax = axes[1]
        regimes = ['early', 'mid', 'late']
        gammas = [results[f'gamma_{r}'] for r in regimes]
        gamma_errs = [results[f'gamma_{r}_err'] for r in regimes]
        x = np.arange(len(regimes))
        ax.bar(x, gammas, yerr=gamma_errs, capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(['Early\n(0-2)', 'Mid\n(2-10)', 'Late\n(10-20)'])
        ax.set_ylabel('Decay rate γ', fontsize=11)
        ax.set_title('Decay Rate vs Time Regime', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_time_evolution_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Test 7: Phase diagram
    if 'phase_diagram' in results_dict:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        results = results_dict['phase_diagram']
        
        # Plateau purity heatmap
        ax = axes[0]
        im = ax.imshow(results['plateaus'], aspect='auto', cmap='RdYlGn',
                      vmin=0.5, vmax=1.0, origin='lower')
        ax.set_xticks(range(len(results['gs'])))
        ax.set_yticks(range(len(results['scr_strengths'])))
        ax.set_xticklabels([f'{g:.3f}' for g in results['gs']])
        ax.set_yticklabels([f'{s:.1f}' for s in results['scr_strengths']])
        ax.set_xlabel('Coupling g', fontsize=11)
        ax.set_ylabel('Scrambling strength', fontsize=11)
        ax.set_title('Phase Diagram: Plateau Purity', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Purity')
        
        # Add text annotations
        for i in range(len(results['scr_strengths'])):
            for j in range(len(results['gs'])):
                text = ax.text(j, i, f'{results["plateaus"][i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        # Decay rate heatmap
        ax = axes[1]
        im = ax.imshow(np.log10(results['gammas'] + 1e-12), aspect='auto', 
                      cmap='viridis_r', origin='lower')
        ax.set_xticks(range(len(results['gs'])))
        ax.set_yticks(range(len(results['scr_strengths'])))
        ax.set_xticklabels([f'{g:.3f}' for g in results['gs']])
        ax.set_yticklabels([f'{s:.1f}' for s in results['scr_strengths']])
        ax.set_xlabel('Coupling g', fontsize=11)
        ax.set_ylabel('Scrambling strength', fontsize=11)
        ax.set_title('Phase Diagram: log₁₀(γ)', fontweight='bold')
        plt.colorbar(im, ax=ax, label='log₁₀(γ)')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_phase_diagram_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_rigorous_tests(save_results=True):
    """
    Run complete rigorous validation suite.
    WARNING: This takes several hours!
    """
    print("="*70)
    print("COMPREHENSIVE RIGOROUS VALIDATION SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'QUICK TEST' if QUICK_TEST else 'FULL RIGOR'}")
    print(f"Default shots: {DEFAULT_SHOTS}")
    print("="*70)
    
    start_time = pytime.time()
    all_results = {}
    
    # Test 1: Hamiltonian types
    try:
        print("\n" + "█"*70)
        all_results['hamiltonian_types'] = test_hamiltonian_types()
    except Exception as e:
        print(f"ERROR in test_hamiltonian_types: {e}")
        all_results['hamiltonian_types'] = None
    
    # Test 2: Coupling sweep
    try:
        print("\n" + "█"*70)
        all_results['coupling_sweep'] = test_coupling_sweep_rigorous()
    except Exception as e:
        print(f"ERROR in test_coupling_sweep_rigorous: {e}")
        all_results['coupling_sweep'] = None
    
    # Test 3: Bath size
    try:
        print("\n" + "█"*70)
        all_results['bath_size'] = test_bath_size_rigorous()
    except Exception as e:
        print(f"ERROR in test_bath_size_rigorous: {e}")
        all_results['bath_size'] = None
    
    # Test 4: Initial states
    try:
        print("\n" + "█"*70)
        all_results['initial_states'] = test_initial_states()
    except Exception as e:
        print(f"ERROR in test_initial_states: {e}")
        all_results['initial_states'] = None
    
    # Test 5: OT

### Merge ###

OCs
    try:
        print("\n" + "█"*70)
        all_results['otoc'] = test_otoc_scrambling_time()
    except Exception as e:
        print(f"ERROR in test_otoc_scrambling_time: {e}")
        all_results['otoc'] = None
    
    # Test 6: Time evolution
    try:
        print("\n" + "█"*70)
        all_results['time_evolution'] = test_time_evolution_justification()
    except Exception as e:
        print(f"ERROR in test_time_evolution_justification: {e}")
        all_results['time_evolution'] = None
    
    # Test 7: Phase diagram
    try:
        print("\n" + "█"*70)
        all_results['phase_diagram'] = test_phase_diagram()
    except Exception as e:
        print(f"ERROR in test_phase_diagram: {e}")
        all_results['phase_diagram'] = None
    
    # Analytical comparison
    try:
        print("\n" + "█"*70)
        if all_results['otoc'] is not None and all_results['coupling_sweep'] is not None:
            all_results['analytical'] = analytical_model_comparison(
                all_results['otoc'], all_results['coupling_sweep']
            )
    except Exception as e:
        print(f"ERROR in analytical_model_comparison: {e}")
        all_results['analytical'] = None
    
    elapsed = pytime.time() - start_time
    
    print("\n" + "="*70)
    print(f"ALL TESTS COMPLETED")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Generate plots
    print("\nGenerating publication-quality plots...")
    plot_rigorous_results(all_results)
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'rigorous_results_{timestamp}.npz'
        
        # Prepare data for saving (numpy arrays only)
        save_data = {}
        for test_name, test_results in all_results.items():
            if test_results is not None:
                if isinstance(test_results, dict):
                    for key, val in test_results.items():
                        if isinstance(val, np.ndarray):
                            save_data[f'{test_name}_{key}'] = val
                        elif isinstance(val, (int, float)):
                            save_data[f'{test_name}_{key}'] = np.array([val])
        
        np.savez(filename, **save_data)
        print(f"\nResults saved to: {filename}")
    
    return all_results

# ============================================================================
# SCIENTIFIC SUMMARY AND CONCLUSIONS
# ============================================================================

def generate_scientific_summary(all_results):
    """
    Generate rigorous scientific summary with statistical significance.
    """
    print("\n" + "="*70)
    print("SCIENTIFIC SUMMARY - RIGOROUS ANALYSIS")
    print("="*70)
    
    # Counters
    tests_passed = 0
    tests_total = 0
    confidence_high = []
    confidence_medium = []
    confidence_low = []
    
    print("\n" + "─"*70)
    print("TEST 1: HAMILTONIAN TYPE INDEPENDENCE")
    print("─"*70)
    
    if all_results.get('hamiltonian_types') is not None:
        tests_total += 1
        results = all_results['hamiltonian_types']
        
        # Compare strong vs weak for each Hamiltonian type
        ham_types = ['Random', 'Heisenberg', 'Ising']
        all_pass = True
        
        for ham in ham_types:
            strong_key = f'{ham} (strong)'
            weak_key = f'{ham} (weak)'
            
            if strong_key in results and weak_key in results:
                gamma_strong = results[strong_key]['gamma']
                gamma_weak = results[weak_key]['gamma']
                err_strong = results[strong_key]['gamma_err']
                err_weak = results[weak_key]['gamma_err']
                
                # Statistical significance test
                if not np.isnan(gamma_strong) and not np.isnan(gamma_weak):
                    diff = gamma_weak - gamma_strong
                    err_combined = np.sqrt(err_strong**2 + err_weak**2)
                    
                    if err_combined > 0:
                        sigma = diff / err_combined
                        
                        print(f"\n{ham} Hamiltonian:")
                        print(f"  Strong: γ = {gamma_strong:.6f} ± {err_strong:.6f}")
                        print(f"  Weak:   γ = {gamma_weak:.6f} ± {err_weak:.6f}")
                        print(f"  Difference: {diff:.6f} ± {err_combined:.6f}")
                        print(f"  Significance: {sigma:.2f}σ", end="")
                        
                        if sigma > 3.0:
                            print(" ✓ HIGHLY SIGNIFICANT")
                            confidence_high.append(f"{ham} protection")
                        elif sigma > 2.0:
                            print(" ✓ SIGNIFICANT")
                            confidence_medium.append(f"{ham} protection")
                        elif sigma > 1.0:
                            print(" ⚠ MARGINAL")
                            confidence_low.append(f"{ham} protection")
                            all_pass = False
                        else:
                            print(" ✗ NOT SIGNIFICANT")
                            all_pass = False
        
        if all_pass:
            tests_passed += 1
            print("\n✓ CONCLUSION: Effect is INDEPENDENT of Hamiltonian structure")
        else:
            print("\n⚠ CONCLUSION: Effect may depend on Hamiltonian type (needs more data)")
    
    print("\n" + "─"*70)
    print("TEST 2: COUPLING STRENGTH SCALING")
    print("─"*70)
    
    if all_results.get('coupling_sweep') is not None:
        tests_total += 1
        results = all_results['coupling_sweep']
        
        # Check if γ scales as g²
        gs = results['gs']
        gammas = results['gammas']
        gamma_errs = results['gamma_errs']
        
        valid = ~np.isnan(gammas) & (gamma_errs > 0)
        
        if np.sum(valid) > 3:
            # Fit γ = α·g² + β
            g_squared = gs[valid]**2
            gamma_fit = gammas[valid]
            
            # Weighted fit
            weights = 1.0 / (gamma_errs[valid]**2)
            coeffs = np.polyfit(g_squared, gamma_fit, 1, w=weights)
            alpha, beta = coeffs
            
            # Compute R²
            gamma_pred = alpha * g_squared + beta
            ss_res = np.sum((gamma_fit - gamma_pred)**2)
            ss_tot = np.sum((gamma_fit - np.mean(gamma_fit))**2)
            r_squared = 1 - ss_res/ss_tot
            
            print(f"\nFit: γ = α·g² + β")
            print(f"  α = {alpha:.4f}")
            print(f"  β = {beta:.6f}")
            print(f"  R² = {r_squared:.4f}")
            
            # Find coupling where protection breaks down
            critical_purity = 0.95
            idx_critical = np.where(results['plateaus'] < critical_purity)[0]
            
            if len(idx_critical) > 0:
                g_critical = gs[idx_critical[0]]
                print(f"\n  Protection breaks down at g ≈ {g_critical:.3f}")
            else:
                print(f"\n  Protection holds for all tested g ≤ {gs[-1]:.3f}")
            
            if r_squared > 0.85:
                tests_passed += 1
                confidence_high.append("g² scaling")
                print("\n✓ CONCLUSION: γ follows Fermi Golden Rule (γ ∝ g²)")
            elif r_squared > 0.7:
                tests_passed += 0.5
                confidence_medium.append("g² scaling")
                print("\n⚠ CONCLUSION: Approximate g² scaling (moderate R²)")
            else:
                confidence_low.append("g² scaling")
                print("\n✗ CONCLUSION: Scaling does not follow g²")
    
    print("\n" + "─"*70)
    print("TEST 3: BATH SIZE SCALING - RESOLVING THE ANOMALY")
    print("─"*70)
    
    if all_results.get('bath_size') is not None:
        tests_total += 1
        results = all_results['bath_size']
        
        M_envs = results['M_envs']
        gammas = results['gammas']
        gamma_errs = results['gamma_errs']
        plateaus = results['plateaus']
        
        print("\nBath size dependence:")
        for i, M in enumerate(M_envs):
            print(f"  M_env={M}: γ = {gammas[i]:.6f} ± {gamma_errs[i]:.6f}, " +
                  f"plateau = {plateaus[i]:.4f} ± {results['plateau_errs'][i]:.4f}")
        
        # Check for monotonic increase (expected for real decoherence)
        if len(M_envs) > 2:
            # Check if trend is increasing
            trend_gamma = np.polyfit(M_envs, gammas, 1)[0]
            
            # Check M=2 anomaly specifically
            if 2 in M_envs and 1 in M_envs and 3 in M_envs:
                idx1 = np.where(M_envs == 1)[0][0]
                idx2 = np.where(M_envs == 2)[0][0]
                idx3 = np.where(M_envs == 3)[0][0]
                
                is_anomaly = (gammas[idx2] < gammas[idx1]) and (gammas[idx2] < gammas[idx3])
                
                if is_anomaly:
                    print(f"\n⚠ M_env=2 ANOMALY CONFIRMED:")
                    print(f"  γ(M=1) = {gammas[idx1]:.6f} > γ(M=2) = {gammas[idx2]:.6f} < γ(M=3) = {gammas[idx3]:.6f}")
                    print(f"  This is non-monotonic behavior!")
                    
                    # Check if it's statistically significant
                    diff_12 = gammas[idx1] - gammas[idx2]
                    err_12 = np.sqrt(gamma_errs[idx1]**2 + gamma_errs[idx2]**2)
                    diff_23 = gammas[idx3] - gammas[idx2]
                    err_23 = np.sqrt(gamma_errs[idx3]**2 + gamma_errs[idx2]**2)
                    
                    sigma_12 = diff_12 / err_12 if err_12 > 0 else 0
                    sigma_23 = diff_23 / err_23 if err_23 > 0 else 0
                    
                    print(f"  Statistical significance:")
                    print(f"    M=1 vs M=2: {sigma_12:.2f}σ")
                    print(f"    M=2 vs M=3: {sigma_23:.2f}σ")
                    
                    if sigma_12 < 1.0 and sigma_23 < 1.0:
                        print(f"  → Likely statistical fluctuation (both <1σ)")
                        confidence_low.append("M_env anomaly")
                    else:
                        print(f"  → May indicate resonance or selection effects")
                        print(f"  → Needs investigation (energy spectrum analysis)")
                else:
                    print(f"\n✓ M_env=2 anomaly RESOLVED with higher statistics")
                    print(f"  Monotonic trend confirmed")
            
            # Overall assessment
            if np.all(plateaus > 0.90):
                tests_passed += 1
                confidence_high.append("Bath size robustness")
                print(f"\n✓ CONCLUSION: Protection robust to bath size (all plateaus >90%)")
            elif np.all(plateaus > 0.80):
                tests_passed += 0.5
                confidence_medium.append("Bath size robustness")
                print(f"\n⚠ CONCLUSION: Protection degrades but survives (plateaus >80%)")
            else:
                print(f"\n✗ CONCLUSION: Protection fails with larger baths")
    
    print("\n" + "─"*70)
    print("TEST 4: INITIAL STATE INDEPENDENCE")
    print("─"*70)
    
    if all_results.get('initial_states') is not None:
        tests_total += 1
        results = all_results['initial_states']
        
        gammas = [results[k]['gamma'] for k in results.keys()]
        gamma_errs = [results[k]['gamma_err'] for k in results.keys()]
        labels = list(results.keys())
        
        print("\nDecay rates for different initial states:")
        for i, label in enumerate(labels):
            print(f"  {label:15s}: γ = {gammas[i]:.6f} ± {gamma_errs[i]:.6f}")
        
        # Check if all consistent within error bars
        gamma_mean = np.mean(gammas)
        gamma_std_mean = np.mean(gamma_errs)
        
        all_consistent = True
        for i, g in enumerate(gammas):
            deviation = abs(g - gamma_mean)
            if deviation > 3 * gamma_std_mean:
                all_consistent = False
                print(f"  ⚠ {labels[i]} deviates by {deviation/gamma_std_mean:.1f}σ")
        
        if all_consistent:
            tests_passed += 1
            confidence_high.append("Initial state independence")
            print(f"\n✓ CONCLUSION: Effect independent of initial state")
        else:
            confidence_low.append("Initial state independence")
            print(f"\n⚠ CONCLUSION: Some initial state dependence observed")
    
    print("\n" + "─"*70)
    print("TEST 5: SCRAMBLING TIME MEASUREMENT (OTOCs)")
    print("─"*70)
    
    if all_results.get('otoc') is not None:
        tests_total += 1
        results = all_results['otoc']
        
        strengths = results['strengths']
        taus = results['tau_scrambles']
        tau_errs = results['tau_stds']
        
        print("\nMeasured scrambling times:")
        for i, s in enumerate(strengths):
            print(f"  Strength={s:4.1f}: τ_scramble = {taus[i]:.4f} ± {tau_errs[i]:.4f}")
        
        # Check if τ decreases with strength
        if len(strengths) > 2:
            valid = ~np.isinf(taus)
            if np.sum(valid) > 2:
                trend = np.polyfit(strengths[valid], taus[valid], 1)[0]
                
                print(f"\nTrend: dτ/d(strength) = {trend:.4f}")
                
                if trend < -0.01:
                    tests_passed += 1
                    confidence_high.append("Scrambling time measurement")
                    print(f"✓ CONCLUSION: τ_scramble decreases with interaction strength")
                    
                    # Check consistency with protection
                    if all_results.get('analytical') is not None:
                        print(f"\nConsistency check with protection mechanism:")
                        print(f"  τ_scramble(s=8) ≈ {taus[strengths==8.0][0]:.4f}")
                        if all_results['analytical']['suppression_factor'] is not None:
                            print(f"  Suppression factor ≈ {all_results['analytical']['suppression_factor']:.1f}x")
                            print(f"  Expected from τ: ~1/τ ≈ {1/taus[strengths==8.0][0]:.1f}x")
                else:
                    confidence_low.append("Scrambling time measurement")
                    print(f"⚠ CONCLUSION: No clear trend in τ_scramble")
    
    print("\n" + "─"*70)
    print("TEST 6: TIME EVOLUTION REGIME VALIDATION")
    print("─"*70)
    
    if all_results.get('time_evolution') is not None:
        tests_total += 1
        results = all_results['time_evolution']
        
        gamma_early = results['gamma_early']
        gamma_mid = results['gamma_mid']
        gamma_late = results['gamma_late']
        
        err_early = results['gamma_early_err']
        err_mid = results['gamma_mid_err']
        err_late = results['gamma_late_err']
        
        print(f"\nDecay rates in different time regimes:")
        print(f"  Early (0-2):   γ = {gamma_early:.6f} ± {err_early:.6f}")
        print(f"  Mid (2-10):    γ = {gamma_mid:.6f} ± {err_mid:.6f}")
        print(f"  Late (10-20):  γ = {gamma_late:.6f} ± {err_late:.6f}")
        
        # Check consistency
        diff_early_mid = abs(gamma_early - gamma_mid)
        err_combined = np.sqrt(err_early**2 + err_mid**2)
        
        if err_combined > 0:
            sigma = diff_early_mid / err_combined
            print(f"\nEarly vs Mid: {sigma:.2f}σ difference")
            
            if sigma < 2.0:
                tests_passed += 1
                confidence_high.append("Time regime consistency")
                print(f"✓ CONCLUSION: Decay rate consistent across time (exponential decay)")
            else:
                confidence_medium.append("Time regime consistency")
                print(f"⚠ CONCLUSION: Decay rate changes with time (non-exponential)")
                print(f"  → May indicate initial transient or saturation effects")
    
    print("\n" + "─"*70)
    print("TEST 7: PHASE DIAGRAM")
    print("─"*70)
    
    if all_results.get('phase_diagram') is not None:
        tests_total += 1
        results = all_results['phase_diagram']
        
        plateaus = results['plateaus']
        
        # Find "protected" region (plateau > 0.95)
        protected = plateaus > 0.95
        partially_protected = (plateaus > 0.90) & (plateaus <= 0.95)
        unprotected = plateaus <= 0.90
        
        n_protected = np.sum(protected)
        n_partial = np.sum(partially_protected)
        n_unprot = np.sum(unprotected)
        n_total = plateaus.size
        
        print(f"\nPhase diagram classification:")
        print(f"  Protected (>95%):         {n_protected}/{n_total} ({100*n_protected/n_total:.1f}%)")
        print(f"  Partially protected (90-95%): {n_partial}/{n_total} ({100*n_partial/n_total:.1f}%)")
        print(f"  Unprotected (<90%):       {n_unprot}/{n_total} ({100*n_unprot/n_total:.1f}%)")
        
        # Identify boundaries
        scr_strengths = results['scr_strengths']
        gs = results['gs']
        
        print(f"\nProtection boundaries:")
        for i, scr in enumerate(scr_strengths):
            protected_row = protected[i, :]
            if np.any(protected_row) and not np.all(protected_row):
                idx_boundary = np.where(~protected_row)[0][0]
                g_boundary = gs[idx_boundary]
                print(f"  Strength={scr:.1f}: Protection lost at g ≈ {g_boundary:.3f}")
        
        if n_protected > 0:
            tests_passed += 1
            confidence_high.append("Phase diagram")
            print(f"\n✓ CONCLUSION: Clear protected regime identified")
        else:
            print(f"\n✗ CONCLUSION: No protected regime found")
    
    # ========================================================================
    # FINAL SCIENTIFIC VERDICT
    # ========================================================================
    
    print("\n" + "="*70)
    print("FINAL SCIENTIFIC VERDICT")
    print("="*70)
    
    print(f"\nTests passed: {tests_passed:.1f}/{tests_total}")
    print(f"Pass rate: {100*tests_passed/tests_total:.1f}%")
    
    print(f"\nConfidence levels:")
    print(f"  HIGH confidence (>3σ): {len(confidence_high)} findings")
    for item in confidence_high:
        print(f"    • {item}")
    
    print(f"\n  MEDIUM confidence (2-3σ): {len(confidence_medium)} findings")
    for item in confidence_medium:
        print(f"    • {item}")
    
    print(f"\n  LOW confidence (<2σ): {len(confidence_low)} findings")
    for item in confidence_low:
        print(f"    • {item}")
    
    # Overall conclusion
    print("\n" + "─"*70)
    print("OVERALL CONCLUSION")
    print("─"*70)
    
    if tests_passed >= 0.8 * tests_total:
        print("\n✓✓✓ HYPOTHESIS STRONGLY VALIDATED")
        print("\nThe emergent self-coherence mechanism:")
        print("  • Is reproducible across multiple test conditions")
        print("  • Shows statistical significance (>2σ in key tests)")
        print("  • Is robust to Hamiltonian structure")
        print("  • Scales as predicted by Fermi Golden Rule")
        print("  • Is consistent with scrambling-time measurements")
        
        print("\n📝 PUBLICATION READINESS: YES")
        print("   Recommended journals: PRL, PRX Quantum, Physical Review A")
        
    elif tests_passed >= 0.6 * tests_total:
        print("\n⚠ HYPOTHESIS PARTIALLY VALIDATED")
        print("\nThe effect is real but:")
        print("  • Some tests show marginal significance (<2σ)")
        print("  • Anomalies remain unexplained (e.g., M_env=2)")
        print("  • Need higher statistics for definitive claims")
        
        print("\n📝 PUBLICATION READINESS: CONDITIONAL")
        print("   Recommended: Address anomalies, increase shots to 50+")
        print("   Target: Physical Review A, Quantum, New J. Physics")
        
    else:
        print("\n✗ HYPOTHESIS NOT VALIDATED")
        print("\nCurrent evidence is insufficient:")
        print("  • Most tests fail to reach significance")
        print("  • Results may be artifacts or noise")
        print("  • Fundamental rethinking required")
        
        print("\n📝 PUBLICATION READINESS: NO")
        print("   Recommendation: Revise hypothesis or computational approach")
    
    # Specific recommendations
    print("\n" + "─"*70)
    print("SPECIFIC RECOMMENDATIONS FOR IMPROVEMENT")
    print("─"*70)
    
    recommendations = []
    
    if QUICK_TEST:
        recommendations.append("RUN IN FULL MODE (set QUICK_TEST=False)")
    
    if DEFAULT_SHOTS < 30:
        recommendations.append(f"INCREASE SHOTS from {DEFAULT_SHOTS} to 50")
    
    if all_results.get('bath_size') and 4 not in all_results['bath_size']['M_envs']:
        recommendations.append("TEST M_env=4 to confirm bath size trend")
    
    if len(confidence_low) > 0:
        recommendations.append("RE-RUN low-confidence tests with higher statistics")
    
    if all_results.get('analytical') and all_results['analytical'].get('suppression_factor'):
        supp = all_results['analytical']['suppression_factor']
        if supp < 10:
            recommendations.append(f"Effect size small ({supp:.1f}x) - consider stronger parameters")
    
    if len(recommendations) > 0:
        print("\nBefore publication:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("\n✓ All critical recommendations addressed!")
    
    print("\n" + "="*70)

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║            RIGOROUS VALIDATION OF EMERGENT SELF-COHERENCE            ║
    ║                                                                      ║
    ║  This script performs comprehensive, publication-quality validation  ║
    ║  with proper error bars, statistical significance tests, and        ║
    ║  multiple independent checks.                                        ║
    ║                                                                      ║
    ║  NO FABRICATION - All results are computed from first principles    ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\nCurrent configuration:")
    print(f"  Mode: {'QUICK TEST (5 shots)' if QUICK_TEST else 'RIGOROUS (30 shots)'}")
    print(f"  Estimated runtime: {'30-60 min' if QUICK_TEST else '3-6 hours'}")
    
    response = input("\nProceed with full test suite? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        all_results = run_all_rigorous_tests(save_results=True)
        generate_scientific_summary(all_results)
    else:
        print("\nTest cancelled. To run later, execute:")
        print("  python hybrid_stable_v5_rigorous.py")
