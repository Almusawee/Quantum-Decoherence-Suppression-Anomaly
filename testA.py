
#!/usr/bin/env python3
"""
TEST A: ENERGY CONTROL - THE CRITICAL MECHANISM TEST
=====================================================

This test determines whether the 7x suppression is due to:
1. Energy scale only (known physics) → ratio ≈ 1.0
2. Scrambling-specific (potentially novel) → ratio > 1.3

Setup:
- Chaotic Hamiltonian (s=8, scrambling)
- Integrable Hamiltonian (Heisenberg, no long-range)
- BOTH matched to same energy scale
- Measure decay rates for each
- Compare statistical significance

Runtime: ~2 hours (100 shots total)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import ttest_ind, linregress
from numpy.random import default_rng
import time as pytime
from datetime import datetime

# ============================================================================
# CONFIGURATION - MATCHED TO YOUR 7x BASELINE
# ============================================================================

N_SYSTEM = 6              # 6 qubits (same as baseline)
M_ENV = 2                 # 2-qubit environment
COUPLING_G = 0.02         # Same coupling as baseline
T_MAX = 10.0              # Same time range
N_STEPS = 50              # Same time resolution
SHOTS_PER_TEST = 50       # 50 shots per Hamiltonian type
SEED_BASE = 5000          # Fixed seed base for reproducibility

# ============================================================================
# QUANTUM MECHANICS HELPERS (from your baseline script)
# ============================================================================

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
    """Random chaotic Hamiltonian (for scrambling)."""
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

def heisenberg_hamiltonian(N, J=1.0, h=0.0, pbc=False):
    """Integrable reference (Heisenberg chain, non-scrambling)."""
    d = 2**N
    H = np.zeros((d, d), dtype=complex)

    # Nearest-neighbor interactions (nearest-neighbor only - no long-range)
    for i in range(N-1):
        H += J * two_on(N, sx, i, sx, i+1)
        H += J * two_on(N, sy, i, sy, i+1)
        H += J * two_on(N, sz, i, sz, i+1)

    # Optional periodic boundary condition
    if pbc and N > 1:
        H += J * two_on(N, sx, N-1, sx, 0)
        H += J * two_on(N, sy, N-1, sy, 0)
        H += J * two_on(N, sz, N-1, sz, 0)

    # Magnetic field
    for i in range(N):
        H += h * op_on(N, sz, i)

    return make_hermitian(H)

# ============================================================================
# MEASUREMENT FUNCTION
# ============================================================================

def measure_decay_rate(N, M_env, H_sys, coupling_g, t_max=10.0, n_steps=50, seed=None):
    """
    Single trajectory measurement.
    Returns: times, purities
    """
    if seed is not None:
        rng = default_rng(seed)
    else:
        rng = default_rng()

    d_sys = 2**N
    d_env = 2**M_env

    # Build total Hamiltonian
    H_env = random_scrambling_H(M_env, strength=0.5, seed=seed+1 if seed else None)
    Htot = np.kron(H_sys, np.eye(d_env, dtype=complex)) + \
           np.kron(np.eye(d_sys, dtype=complex), H_env)

    # Coupling term
    n_boundary = max(1, int(N * 0.3))
    A_sys = np.zeros((d_sys, d_sys), dtype=complex)
    for b in range(n_boundary):
        A_sys += op_on(N, sz, b)
    A_sys = make_hermitian(A_sys)

    B_env = np.zeros((d_env, d_env), dtype=complex)
    for e in range(M_env):
        B_env += op_on(M_env, sz, e)
    B_env = make_hermitian(B_env)

    H_int = coupling_g * np.kron(A_sys, B_env)
    Htot = Htot + H_int

    # Initial state
    psi_sys = rng.normal(size=d_sys) + 1j * rng.normal(size=d_sys)
    psi_sys /= np.linalg.norm(psi_sys)
    rho_S0 = np.outer(psi_sys, psi_sys.conj())
    rho_E0 = np.eye(d_env) / d_env
    rho0 = np.kron(rho_S0, rho_E0)

    # Evolution
    times = np.linspace(0, t_max, n_steps)
    purities = []

    for t in times:
        U = expm(-1j * Htot * t)
        rho_t = U @ rho0 @ U.conj().T
        rhoS = partial_trace_env(rho_t, N, M_env)
        rhoS = normalize_rho(rhoS)
        purities.append(purity(rhoS))

    return times, np.array(purities)

def fit_exponential_decay(times, purities_mean, purities_std, fit_range=(0, 2)):
    """Fit P(t) = exp(-γt) with error estimation. Robust to small decay."""

    mask = (times >= fit_range[0]) & (times <= fit_range[1])
    t_fit = times[mask]
    p_mean = np.clip(purities_mean[mask], 1e-12, 1.0)
    p_std = np.clip(purities_std[mask], 1e-8, 1.0)  # Avoid division by zero

    if len(t_fit) < 3:
        return np.nan, np.nan, np.nan

    try:
        # Fit log(P) = log(P0) - γt
        # Use weights, but protect against division by zero
        weights = np.where(p_std > 1e-8, 1.0 / (p_std**2), 1e-8)
        weights = np.clip(weights, 1e-8, 1e8)  # Clip extreme weights

        coeffs = np.polyfit(t_fit, np.log(p_mean), 1, w=weights)
        gamma = -coeffs[0]

        # If gamma is negative or zero, system isn't decaying
        if gamma <= 0:
            gamma = 1e-10  # Minimum detectable decay

        # Bootstrap error
        gammas_boot = []
        for _ in range(100):
            p_sample = np.clip(p_mean + np.random.normal(0, p_std), 1e-12, 1.0)
            try:
                coeffs_boot = np.polyfit(t_fit, np.log(p_sample), 1)
                gamma_boot = -coeffs_boot[0]
                if gamma_boot > 0:  # Only include positive decay rates
                    gammas_boot.append(gamma_boot)
            except:
                pass

        if len(gammas_boot) > 2:
            gamma_err = np.std(gammas_boot)
        else:
            gamma_err = np.abs(gamma) * 0.3  # Fallback: 30% error

        # R² value
        p_pred = np.exp(coeffs[1] - gamma * t_fit)
        ss_res = np.sum((p_mean - p_pred)**2)
        ss_tot = np.sum((p_mean - np.mean(p_mean))**2)

        if ss_tot > 1e-10:
            r_squared = max(-1, min(1, 1 - ss_res / ss_tot))  # Clip to [-1, 1]
        else:
            r_squared = np.nan

        return gamma, gamma_err, r_squared

    except Exception as e:
        return np.nan, np.nan, np.nan

# ============================================================================
# ANALYTICAL PREDICTIONS
# ============================================================================

def analytical_predictions():
    """
    Show what theory predicts BEFORE running the test.
    """

    print("\n" + "="*80)
    print("ANALYTICAL PREDICTIONS (Theory)")
    print("="*80)

    print("""
QUANTUM ZENO EFFECT THEORY:
===========================

For system-environment coupling: H_int = g * A_sys ⊗ B_env

The decoherence rate scales as:
  γ ∝ g² * τ_c / (1 + ω² τ_c²)

where:
  g = coupling strength = 0.02
  τ_c = environment correlation time ~ 0.1-1
  ω = system energy scale

KEY PREDICTION:
If two Hamiltonians have the SAME energy scale ω (both norm-matched),
then Zeno theory predicts they have the SAME decoherence rate.

CHAOTIC vs INTEGRABLE at SAME ω:
  • Chaotic: H = Σ_i h_i σ_i + Σ_{i<j} J_ij σ_i σ_j  (all-to-all)
            → Fast information spreading
            → Strong Zeno effect

  • Integrable: H = Σ_i J σ_x^i σ_x^{i+1} + ...  (nearest-neighbor)
               → Slow information spreading
               → Similar energy scale if norm-matched
               → Should give similar γ if Zeno dominates

THEORETICAL PREDICTIONS FOR TEST A:
====================================

SCENARIO 1: Zeno Effect Dominates (Known Physics)
  Expected: γ_int / γ_chaos ≈ 1.0 ± 0.3
  Interpretation: Energy scale is what matters
  Publication: Measurement of known effect (PRA)

SCENARIO 2: Scrambling Provides Additional Protection (Novel)
  Expected: γ_int / γ_chaos > 1.5
  Interpretation: Chaotic systems protect better than integrable
  Meaning: Scrambling structure itself provides protection
  Publication: Discovery paper (PRL)

SCENARIO 3: Ambiguous (Need More Data)
  Expected: γ_int / γ_chaos = 1.2-1.4, p-value ≈ 0.05-0.1
  Interpretation: Unclear
  Next step: Increase shots to 100 per type

YOUR BASELINE (for reference):
  s=0 (no scrambling): γ = 0.00203 s⁻¹
  s=8 (strong scrambling): γ = 0.000286 s⁻¹
  Suppression: 7.09x

This large suppression could be either:
  • Energy-scale effect (high-ω Hamiltonian at s=8)
  • Scrambling-specific effect
  → TEST A will distinguish these
""")

    print("="*80)
    print("Now running numerical test to compare theory vs reality...")
    print("="*80)

# ============================================================================
# MAIN TEST A FUNCTION
# ============================================================================

def test_a_energy_control():
    """
    TEST A: Energy Control - Critical mechanism test.
    """

    print("\n" + "█"*80)
    print("█" + " TEST A: ENERGY CONTROL (NUMERICAL EXPERIMENT)".center(78) + "█")
    print("█"*80)

    print(f"\nConfiguration (matched to 7x baseline):")
    print(f"  System: N={N_SYSTEM} qubits")
    print(f"  Environment: M={M_ENV} qubits")
    print(f"  Coupling: g={COUPLING_G}")
    print(f"  Shots per type: {SHOTS_PER_TEST}")
    print(f"  Total trajectories: {2*SHOTS_PER_TEST}")
    print(f"  Expected runtime: ~2 hours\n")

    start_time = pytime.time()

    # ========================================================================
    # STEP 1: BUILD HAMILTONIANS
    # ========================================================================

    print("="*80)
    print("STEP 1: Building Hamiltonians")
    print("="*80)

    print("\nGenerating CHAOTIC Hamiltonian (s=8, with long-range)...")
    H_chaos = random_scrambling_H(N_SYSTEM, strength=8.0, seed=SEED_BASE)
    eigvals_chaos = np.linalg.eigvalsh(H_chaos)
    omega_chaos = np.std(eigvals_chaos)
    norm_chaos = np.linalg.norm(H_chaos, 'fro')

    print(f"  Energy scale (std of eigenvalues): ω_chaos = {omega_chaos:.4f}")
    print(f"  Frobenius norm: ||H_chaos|| = {norm_chaos:.4f}")

    print("\nGenerating INTEGRABLE Hamiltonian (Heisenberg, nearest-neighbor only)...")
    H_integrable = heisenberg_hamiltonian(N_SYSTEM, J=1.0, h=0.5, pbc=False)
    norm_int_original = np.linalg.norm(H_integrable, 'fro')

    # Scale integrable to match norm of chaotic
    H_integrable = H_integrable * (norm_chaos / norm_int_original)

    eigvals_int = np.linalg.eigvalsh(H_integrable)
    omega_int = np.std(eigvals_int)
    norm_int = np.linalg.norm(H_integrable, 'fro')

    print(f"  Energy scale (std of eigenvalues): ω_int = {omega_int:.4f}")
    print(f"  Frobenius norm (after scaling): ||H_int|| = {norm_int:.4f}")

    print("\n" + "-"*80)
    print("ENERGY SCALE VERIFICATION:")
    print("-"*80)
    print(f"ω_chaos = {omega_chaos:.4f}")
    print(f"ω_int   = {omega_int:.4f}")
    print(f"Ratio (ω_int / ω_chaos) = {omega_int/omega_chaos:.3f}")
    print(f"||H_chaos|| = {norm_chaos:.4f}")
    print(f"||H_int||   = {norm_int:.4f}")
    print(f"Ratio (||H_int|| / ||H_chaos||) = {norm_int/norm_chaos:.3f}")

    if abs(norm_int - norm_chaos) / norm_chaos < 0.01:
        print("\n✓ Norms well matched (within 1%)")
    else:
        print(f"\n⚠️  Norms differ by {abs(norm_int - norm_chaos) / norm_chaos * 100:.1f}%")

    # ========================================================================
    # STEP 2: MEASURE DECAY RATES
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 2: Measuring Decay Rates")
    print("="*80)

    print(f"\n[1/2] Measuring CHAOTIC ({SHOTS_PER_TEST} shots)...")
    all_p_chaos = []
    for shot in range(SHOTS_PER_TEST):
        times, purities = measure_decay_rate(
            N_SYSTEM, M_ENV, H_chaos, COUPLING_G, T_MAX, N_STEPS,
            seed=SEED_BASE + 100 + shot
        )
        all_p_chaos.append(purities)

        if (shot + 1) % 10 == 0:
            print(f"  {shot+1}/{SHOTS_PER_TEST} complete")

    p_chaos_mean = np.mean(all_p_chaos, axis=0)
    p_chaos_std = np.std(all_p_chaos, axis=0)

    print(f"\n[2/2] Measuring INTEGRABLE ({SHOTS_PER_TEST} shots)...")
    all_p_int = []
    for shot in range(SHOTS_PER_TEST):
        times, purities = measure_decay_rate(
            N_SYSTEM, M_ENV, H_integrable, COUPLING_G, T_MAX, N_STEPS,
            seed=SEED_BASE + 200 + shot
        )
        all_p_int.append(purities)

        if (shot + 1) % 10 == 0:
            print(f"  {shot+1}/{SHOTS_PER_TEST} complete")

    p_int_mean = np.mean(all_p_int, axis=0)
    p_int_std = np.std(all_p_int, axis=0)

    # ========================================================================
    # STEP 3: FIT DECAY RATES
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 3: Fitting Exponential Decay")
    print("="*80)

    gamma_chaos, err_chaos, r2_chaos = fit_exponential_decay(
        times, p_chaos_mean, p_chaos_std, fit_range=(0, 2)
    )
    gamma_int, err_int, r2_int = fit_exponential_decay(
        times, p_int_mean, p_int_std, fit_range=(0, 2)
    )

    print(f"\nChaotic:")
    print(f"  γ_chaos = {gamma_chaos:.6f} ± {err_chaos:.6f} s⁻¹")
    print(f"  R² = {r2_chaos:.3f}")

    print(f"\nIntegrable:")
    print(f"  γ_int = {gamma_int:.6f} ± {err_int:.6f} s⁻¹")
    print(f"  R² = {r2_int:.3f}")

   # ========================================================================
    # STEP 4: STATISTICAL COMPARISON
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 4: Statistical Comparison")
    print("="*80)

    if gamma_chaos > 0 and gamma_int > 0:
        ratio = gamma_int / gamma_chaos
        combined_err = ratio * np.sqrt(
            (err_int/gamma_int)**2 + (err_chaos/gamma_chaos)**2
        )
    else:
        ratio = np.nan
        combined_err = np.nan

    print(f"\nRatio (γ_int / γ_chaos): {ratio:.3f} ± {combined_err:.3f}")

    # T-test on individual trajectories
    gamma_vals_chaos = []
    gamma_vals_int = []

    for i in range(len(all_p_chaos)):
        g_c, _, _ = fit_exponential_decay(
            times, all_p_chaos[i], np.zeros_like(all_p_chaos[i]) + 1e-8,
            fit_range=(0, 2)
        )
        if not np.isnan(g_c) and g_c > 0:
            gamma_vals_chaos.append(g_c)

    for i in range(len(all_p_int)):
        g_i, _, _ = fit_exponential_decay(
            times, all_p_int[i], np.zeros_like(all_p_int[i]) + 1e-8,
            fit_range=(0, 2)
        )
        if not np.isnan(g_i) and g_i > 0:
            gamma_vals_int.append(g_i)

    # T-test only if we have enough samples
    if len(gamma_vals_chaos) > 2 and len(gamma_vals_int) > 2:
        try:
            t_stat, p_value = ttest_ind(gamma_vals_chaos, gamma_vals_int)
            if np.isnan(p_value):
                p_value = 1.0
        except:
            t_stat = np.nan
            p_value = 1.0
    else:
        t_stat = np.nan
        p_value = np.nan
        print(f"\n⚠️  Warning: T-test requires >2 samples per group")
        print(f"  Chaotic: {len(gamma_vals_chaos)} valid measurements")
        print(f"  Integrable: {len(gamma_vals_int)} valid measurements")

    print(f"\nT-test:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")

    # ========================================================================
    # STEP 5: INTERPRETATION
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 5: Interpretation")
    print("="*80)

    print(f"\nDecay rates normalized to chaotic:")
    print(f"  Chaotic baseline: γ_chaos = {gamma_chaos:.6f} s⁻¹")
    print(f"  Integrable: γ_int = {gamma_int:.6f} s⁻¹")
    print(f"  Ratio: {ratio:.2f}x")

    if ratio < 1.2 and p_value > 0.05:
        interpretation = "ENERGY_SCALE_ONLY"
        print(f"\n  → Decay rates are SIMILAR (within 20%, p > 0.05)")
        print(f"  → Suggests mechanism is ENERGY-SCALE dependent")
        print(f"  → Known physics: Quantum Zeno or similar")

    elif ratio > 1.3 and p_value < 0.05:
        interpretation = "SCRAMBLING_SPECIFIC"
        print(f"\n  → Integrable decays FASTER (ratio = {ratio:.2f}x, p < 0.05)")
        print(f"  → Suggests SCRAMBLING provides additional protection")
        print(f"  → Chaotic systems protect more than integrable")

    else:
        interpretation = "AMBIGUOUS"
        print(f"\n  → Result is inconclusive (ratio = {ratio:.2f}, p = {p_value:.3f})")
        print(f"  → Recommendation: Increase shots to 100 per type")
    # ========================================================================
    # STEP 6: PLOTTING
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 6: Generating Plots")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Purity trajectories
    ax = axes[0, 0]
    ax.plot(times, p_chaos_mean, '-', linewidth=2.5, label='Chaotic (scrambling)',
            color='red', alpha=0.8)
    ax.fill_between(times, p_chaos_mean - p_chaos_std, p_chaos_mean + p_chaos_std,
                    alpha=0.2, color='red')

    ax.plot(times, p_int_mean, '-', linewidth=2.5, label='Integrable (non-scrambling)',
            color='blue', alpha=0.8)
    ax.fill_between(times, p_int_mean - p_int_std, p_int_mean + p_int_std,
                    alpha=0.2, color='blue')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Purity P(t)', fontsize=12)
    ax.set_title('Purity Trajectories (50 shots each)', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Log scale
    ax = axes[0, 1]
    valid_chaos = p_chaos_mean > 1e-3
    valid_int = p_int_mean > 1e-3

    ax.semilogy(times[valid_chaos], p_chaos_mean[valid_chaos], 'o-',
                label='Chaotic', color='red', alpha=0.8, markersize=6)
    ax.semilogy(times[valid_int], p_int_mean[valid_int], 's-',
                label='Integrable', color='blue', alpha=0.8, markersize=6)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Purity P(t) [log scale]', fontsize=12)
    ax.set_title('Exponential Decay Check', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Plot 3: Decay rates comparison
    ax = axes[1, 0]
    gammas = [gamma_chaos, gamma_int]
    errs = [err_chaos, err_int]
    labels = ['Chaotic\n(Scrambling)', 'Integrable\n(Non-scrambling)']
    colors = ['red', 'blue']

    bars = ax.bar(range(2), gammas, yerr=errs, capsize=10, color=colors,
                  alpha=0.7, edgecolor='black', linewidth=2, width=0.6)

    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Decay Rate γ (s⁻¹)', fontsize=12)
    ax.set_title('Decay Rate Comparison', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    ax.text(0.5, max(gammas)*1.5, f'Ratio: {ratio:.2f}x\np={p_value:.4f}',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='black', linewidth=2))

    # Plot 4: Interpretation
    ax = axes[1, 1]
    ax.axis('off')

    if interpretation == "ENERGY_SCALE_ONLY":
        verdict_text = "ENERGY-SCALE ONLY\n(Known Physics)\n\nThe 7x suppression\nis due to Zeno effect\nor similar mechanism"
        color = 'orange'
    elif interpretation == "SCRAMBLING_SPECIFIC":
        verdict_text = "SCRAMBLING MATTERS\n(Potentially Novel)\n\nScrambling provides\nadditional protection\nbeyond energy scale"
        color = 'green'
    else:
        verdict_text = "AMBIGUOUS RESULT\n(Need More Data)\n\nIncrease shots\nor test at extreme\nparameters"
        color = 'yellow'

    ax.text(0.5, 0.5, verdict_text, ha='center', va='center', fontsize=14,
            fontweight='bold', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8,
                     edgecolor='black', linewidth=3, pad=1.5))

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f'test_a_energy_control_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved: {plot_file}")
    plt.show()

    # ========================================================================
    # FINAL VERDICT
    # ========================================================================

    elapsed = pytime.time() - start_time

    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    print(f"\nComputation time: {elapsed/3600:.2f} hours")
    print(f"Result: {interpretation}")
    print(f"  Ratio (γ_int / γ_chaos): {ratio:.3f} ± {combined_err:.3f}")
    print(f"  p-value: {p_value:.6f}")

    if interpretation == "ENERGY_SCALE_ONLY":
        print("\nCONCLUSION: 7x suppression explained by energy scale")
        print("  Mechanism: Quantum Zeno Effect or similar (known physics)")
        print("  Publication: Physical Review A (measurement of known effect)")
    elif interpretation == "SCRAMBLING_SPECIFIC":
        print("\nCONCLUSION: Scrambling provides additional protection")
        print("  Mechanism: Potentially novel (beyond Zeno)")
        print("  Publication: Need expert examination")
    else:
        print("\nCONCLUSION: Result inconclusive")
        print("  Recommendation: Run with 100 shots per type")

    print("="*80)

    return {
        'gamma_chaos': gamma_chaos,
        'gamma_int': gamma_int,
        'ratio': ratio,
        'p_value': p_value,
        'interpretation': interpretation,
        'computation_time_hours': elapsed / 3600
    }

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║             TEST A: ENERGY CONTROL - CRITICAL TEST                   ║
    ║                                                                      ║
    ║  This test determines if your 7x suppression is:                     ║
    ║    • Known physics (energy-scale effect) → ratio ≈ 1.0               ║
    ║    • Novel physics (scrambling-specific) → ratio > 1.3               ║
    ║                                                                      ║
    ║  Runtime: ~2 hours (100 total trajectories)                          ║
    ║  This will definitively answer your research question.               ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Show analytical predictions first
    analytical_predictions()

    # Then run the test
    response = input("\nProceed with TEST A? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        print("\n🚀 Starting TEST A...\n")
        results = test_a_energy_control()

        print("\n" + "█"*80)
        print("█" + " TEST A COMPLETE - RESULTS ANALYSIS".center(78) + "█")
        print("█"*80)

        print("\nSUMMARY:")
        print(f"  Interpretation: {results['interpretation']}")
        print(f"  Ratio (γ_int/γ_chaos): {results['ratio']:.3f}")
        print(f"  p-value: {results['p_value']:.6f}")
        print(f"  Computation time: {results['computation_time_hours']:.2f} hours")

        print("\n" + "─"*80)
        print("COMPARISON TO THEORY:")
        print("─"*80)
        print("\nTheory predicted:")
        print("  • If Zeno only: ratio ≈ 1.0-1.2")
        print("  • If scrambling matters: ratio > 1.5")
        print(f"\nYou measured:")
        print(f"  • Ratio = {results['ratio']:.3f}")
        print(f"  • p-value = {results['p_value']:.6f}")

        if results['interpretation'] == 'ENERGY_SCALE_ONLY':
            print(f"\n✓ MATCHES THEORY: Energy-scale dominates")
            print("  Your 7x suppression is explained by Zeno effect")
            print("  → Run TEST B (system size scaling) to confirm")
            print("  → Then publish as PRA paper")
        elif results['interpretation'] == 'SCRAMBLING_SPECIFIC':
            print(f"\n✓ DEVIATES FROM THEORY: Scrambling matters!")
            print("  Chaotic systems protect more than integrable")
            print("  → Run TEST B & C to fully characterize")
            print("  → Develop theoretical model")
            print("  → Publication target: Need expert view first")
        else:
            print(f"\n⚠️  INCONCLUSIVE: Need higher precision")
            print("  → Increase shots to 100 per type and rerun")
            print("  → Or test at more extreme parameters")

        print("\n" + "="*80)

    else:
        print("Test cancelled.")
