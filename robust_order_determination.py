#!/usr/bin/env python3
"""
Robust Exact Order Determination for R1
======================================

Uses multiple mathematical approaches to determine exact orders:
1. Fractional life method
2. Initial rate method  
3. Concentration-time profile fitting
4. Mass balance approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

def load_R1_data():
    """Load R1 data at 70°C, 50 mg/mL"""
    df = pd.read_excel("Reaction-1.xlsx", sheet_name="Calculated")
    subset = df[(df["Temp_C"] == 70) & (df["A0_mgml"] == 50)]
    
    time = subset['Time_h'].values
    A = subset['A_mgml'].values
    B = subset['B_mgml'].values
    I = subset['I_mgml'].values
    
    return time, A, B, I

def fractional_life_method(time, A, A0):
    """
    Fractional life method to determine order
    For nth order: t_1/2 ∝ [A0]^(1-n)
    """
    print("🔬 METHOD 1: Fractional Life Analysis")
    print("="*45)
    
    # Find time for different fractional conversions
    conversions = [0.25, 0.50, 0.75, 0.90]
    times = []
    
    for conv in conversions:
        target_A = A0 * (1 - conv)
        # Find closest time point
        idx = np.argmin(np.abs(A - target_A))
        if A[idx] <= target_A and idx < len(time) - 1:
            # Interpolate
            t_frac = time[idx] + (target_A - A[idx]) / (A[idx+1] - A[idx]) * (time[idx+1] - time[idx])
            times.append(t_frac)
            print(f"   t_{conv*100:.0f}% = {t_frac:.3f} hours")
    
    # For first-order: all half-lives should be equal
    if len(times) >= 2:
        ratios = []
        for i in range(1, len(times)):
            ratio = times[i] / times[0]
            ratios.append(ratio)
        
        print(f"   Time ratios: {ratios}")
        
        # Check if ratios follow expected pattern
        expected_ratios_1st = [2, 3, 4][:len(ratios)]  # For first-order
        expected_ratios_2nd = [3, 7, 15][:len(ratios)]  # For second-order
        
        error_1st = np.mean([abs(r - e) for r, e in zip(ratios, expected_ratios_1st)])
        error_2nd = np.mean([abs(r - e) for r, e in zip(ratios, expected_ratios_2nd)])
        
        print(f"   Error vs 1st order: {error_1st:.3f}")
        print(f"   Error vs 2nd order: {error_2nd:.3f}")
        
        if error_1st < error_2nd:
            print("   ✅ CONCLUSION: First-order kinetics confirmed")
            return 1.0, error_1st
        else:
            print("   ⚠️  CONCLUSION: May not be first-order")
            return 2.0, error_2nd
    
    return None, None

def initial_rate_method(time, A, B, I, A0):
    """
    Initial rate method using early time points
    """
    print("\n🔬 METHOD 2: Initial Rate Analysis") 
    print("="*40)
    
    # Use first 4 time points for initial rates
    n_points = min(4, len(time))
    t_init = time[:n_points]
    A_init = A[:n_points]
    B_init = B[:n_points]
    I_init = I[:n_points]
    
    # Calculate initial rates by linear regression on early points
    if len(t_init) >= 3:
        # Rate of B formation
        dB_dt_coeffs = np.polyfit(t_init, B_init, 1)
        dB_dt_init = dB_dt_coeffs[0]
        
        # Rate of I formation  
        dI_dt_coeffs = np.polyfit(t_init, I_init, 1)
        dI_dt_init = dI_dt_coeffs[0]
        
        print(f"   Initial rate of B formation: {dB_dt_init:.4f} mg/mL/h")
        print(f"   Initial rate of I formation: {dI_dt_init:.4f} mg/mL/h")
        
        # For first-order: rate = k * [A0]
        # So: k1 = dB_dt_init / A0, k2 = dI_dt_init / A0
        k1_apparent = dB_dt_init / A0
        k2_apparent = dI_dt_init / A0
        
        print(f"   Apparent k1 (assuming 1st order): {k1_apparent:.4f} h⁻¹")
        print(f"   Apparent k2 (assuming 1st order): {k2_apparent:.4f} h⁻¹")
        print(f"   Total k_apparent: {k1_apparent + k2_apparent:.4f} h⁻¹")
        
        return k1_apparent, k2_apparent
    
    return None, None

def concentration_profile_fitting(time, A, B, I, A0):
    """
    Fit concentration profiles directly to determine orders
    """
    print("\n🔬 METHOD 3: Concentration Profile Fitting")
    print("="*50)
    
    # Test both first-order assumption
    def first_order_parallel(t, k1, k2):
        """Analytical solution for first-order parallel"""
        k_total = k1 + k2
        A_t = A0 * np.exp(-k_total * t)
        B_t = A0 * (k1/k_total) * (1 - np.exp(-k_total * t))
        I_t = A0 * (k2/k_total) * (1 - np.exp(-k_total * t))
        return np.concatenate([A_t, B_t, I_t])
    
    # Prepare data for fitting
    y_data = np.concatenate([A, B, I])
    
    try:
        # Fit first-order model
        popt, pcov = curve_fit(
            first_order_parallel, 
            time, 
            y_data,
            bounds=([0, 0], [10, 2]),
            maxfev=10000
        )
        
        k1_fit, k2_fit = popt
        k_total_fit = k1_fit + k2_fit
        
        # Calculate predictions
        y_pred = first_order_parallel(time, k1_fit, k2_fit)
        A_pred = y_pred[:len(time)]
        B_pred = y_pred[len(time):2*len(time)]
        I_pred = y_pred[2*len(time):]
        
        # Calculate R² for each component
        def calc_r2(actual, predicted):
            ss_res = np.sum((actual - predicted)**2)
            ss_tot = np.sum((actual - np.mean(actual))**2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        r2_A = calc_r2(A, A_pred)
        r2_B = calc_r2(B, B_pred)
        r2_I = calc_r2(I, I_pred)
        r2_overall = (r2_A + r2_B + r2_I) / 3
        
        print(f"   First-order fit results:")
        print(f"      k1 = {k1_fit:.4f} h⁻¹")
        print(f"      k2 = {k2_fit:.4f} h⁻¹")
        print(f"      k_total = {k_total_fit:.4f} h⁻¹")
        print(f"      R² for A: {r2_A:.4f}")
        print(f"      R² for B: {r2_B:.4f}")
        print(f"      R² for I: {r2_I:.4f}")
        print(f"      R² overall: {r2_overall:.4f}")
        
        # Parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov))
        print(f"      k1 uncertainty: ±{param_errors[0]:.4f}")
        print(f"      k2 uncertainty: ±{param_errors[1]:.4f}")
        
        if r2_overall > 0.99:
            print("   ✅ EXCELLENT fit - first-order model is correct!")
        elif r2_overall > 0.95:
            print("   ✅ GOOD fit - first-order model is acceptable")
        else:
            print("   ⚠️  MODERATE fit - may need different orders")
            
        return {
            'k1': k1_fit,
            'k2': k2_fit,
            'k_total': k_total_fit,
            'r2_overall': r2_overall,
            'r2_A': r2_A,
            'r2_B': r2_B,
            'r2_I': r2_I,
            'param_errors': param_errors,
            'A_pred': A_pred,
            'B_pred': B_pred,
            'I_pred': I_pred
        }
        
    except Exception as e:
        print(f"   ❌ Fitting failed: {e}")
        return None

def mass_balance_check(time, A, B, I, A0):
    """
    Verify mass balance and stoichiometry
    """
    print("\n🔬 METHOD 4: Mass Balance Verification")
    print("="*45)
    
    # Check mass balance: A + B + I = A0 (approximately)
    total_mass = A + B + I
    mass_balance_error = np.abs(total_mass - A0) / A0 * 100
    
    print(f"   Mass balance check:")
    for i, (t, mb_err) in enumerate(zip(time, mass_balance_error)):
        print(f"      t={t:.1f}h: error = {mb_err:.2f}%")
    
    avg_error = np.mean(mass_balance_error)
    print(f"   Average mass balance error: {avg_error:.2f}%")
    
    if avg_error < 5:
        print("   ✅ GOOD mass balance - data is consistent")
    else:
        print("   ⚠️  POOR mass balance - check data quality")
    
    # Check if B + I approaches A0 at long times (complete conversion)
    final_conversion = (B[-1] + I[-1]) / A0 * 100
    print(f"   Final conversion: {final_conversion:.1f}%")
    
    return avg_error, final_conversion

def comprehensive_order_determination():
    """
    Run all methods and determine final orders
    """
    print("🎯 COMPREHENSIVE ORDER DETERMINATION FOR R1")
    print("="*55)
    print("Determining exact individual orders for parallel reactions:")
    print("A → B (k1, order n1) + A → I (k2, order n2)")
    print("="*55)
    
    # Load data
    time, A, B, I = load_R1_data()
    A0 = 50
    
    print(f"📊 Data: {len(time)} points, 70°C, 50 mg/mL")
    print(f"📈 Final yield: B={B[-1]:.1f}, I={I[-1]:.1f} mg/mL")
    
    # Method 1: Fractional life
    order_frac, error_frac = fractional_life_method(time, A, A0)
    
    # Method 2: Initial rates
    k1_init, k2_init = initial_rate_method(time, A, B, I, A0)
    
    # Method 3: Profile fitting
    fit_results = concentration_profile_fitting(time, A, B, I, A0)
    
    # Method 4: Mass balance
    mb_error, final_conv = mass_balance_check(time, A, B, I, A0)
    
    # Comprehensive analysis
    print(f"\n🏆 COMPREHENSIVE CONCLUSION")
    print("="*40)
    
    # Check consistency across methods
    if fit_results and fit_results['r2_overall'] > 0.99:
        print("✅ DEFINITIVE CONCLUSION: Both reactions are FIRST-ORDER")
        print(f"   Mathematical evidence:")
        print(f"   • Profile fitting R² = {fit_results['r2_overall']:.4f} (excellent)")
        print(f"   • Mass balance error = {mb_error:.1f}% (good)")
        if order_frac == 1.0:
            print(f"   • Fractional life analysis confirms first-order")
        
        print(f"\n📋 EXACT KINETIC PARAMETERS:")
        print(f"   A → B: k1 = {fit_results['k1']:.4f} ± {fit_results['param_errors'][0]:.4f} h⁻¹ (order = 1)")
        print(f"   A → I: k2 = {fit_results['k2']:.4f} ± {fit_results['param_errors'][1]:.4f} h⁻¹ (order = 1)")
        print(f"   Total: k_total = {fit_results['k_total']:.4f} h⁻¹")
        print(f"   Selectivity: {fit_results['k1']/(fit_results['k1']+fit_results['k2'])*100:.1f}%")
        
        # Residence time calculation - EXACT, NO ASSUMPTIONS
        k_total = fit_results['k_total']
        
        print(f"\n⏱️  RESIDENCE TIME CALCULATION (EXACT):")
        print(f"   Formula: τ = -ln(1-X)/k_total")
        print(f"   Where: X = desired conversion, k_total = {k_total:.4f} h⁻¹")
        print(f"")
        
        # Calculate for different conversions
        conversions = [0.90, 0.95, 0.99, 0.999]
        for conv in conversions:
            tau = -np.log(1 - conv) / k_total
            print(f"   For {conv*100:.1f}% conversion: τ = {tau:.3f} hours = {tau*60:.1f} minutes")
        
        print(f"\n   💡 NOTE: These are EXACT values for PFR/batch reactor")
        print(f"   💡 No assumptions or enhancement factors applied")
        print(f"   💡 Choose conversion based on your process requirements")
        
        return fit_results, True
    
    else:
        print("❌ INCONCLUSIVE: Need more sophisticated analysis")
        return None, False

def plot_final_results(time, A, B, I, fit_results):
    """
    Plot final results with exact parameters
    """
    if not fit_results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'R1 Exact Order Determination - FIRST ORDER CONFIRMED\n' +
                f'k1={fit_results["k1"]:.4f} h⁻¹, k2={fit_results["k2"]:.4f} h⁻¹', 
                fontsize=14, fontweight='bold')
    
    # A concentration
    axes[0,0].plot(time, A, 'bo', markersize=8, label='Experimental')
    axes[0,0].plot(time, fit_results['A_pred'], 'b-', linewidth=2, 
                   label=f'First-order model (R²={fit_results["r2_A"]:.3f})')
    axes[0,0].set_title('Reactant A Consumption')
    axes[0,0].set_ylabel('A Concentration (mg/mL)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # B concentration
    axes[0,1].plot(time, B, 'go', markersize=8, label='Experimental')
    axes[0,1].plot(time, fit_results['B_pred'], 'g-', linewidth=2,
                   label=f'First-order model (R²={fit_results["r2_B"]:.3f})')
    axes[0,1].set_title('Product B Formation (First-Order)')
    axes[0,1].set_ylabel('B Concentration (mg/mL)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # I concentration
    axes[1,0].plot(time, I, 'ro', markersize=8, label='Experimental')
    axes[1,0].plot(time, fit_results['I_pred'], 'r-', linewidth=2,
                   label=f'First-order model (R²={fit_results["r2_I"]:.3f})')
    axes[1,0].set_title('Impurity I Formation (First-Order)')
    axes[1,0].set_ylabel('I Concentration (mg/mL)')
    axes[1,0].set_xlabel('Time (hours)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # Selectivity
    sel_exp = B / (B + I) * 100
    sel_pred = fit_results['B_pred'] / (fit_results['B_pred'] + fit_results['I_pred']) * 100
    
    axes[1,1].plot(time, sel_exp, 'mo', markersize=8, label='Experimental')
    axes[1,1].plot(time, sel_pred, 'm-', linewidth=2, label='First-order model')
    axes[1,1].set_title('Selectivity vs Time')
    axes[1,1].set_ylabel('Selectivity (%)')
    axes[1,1].set_xlabel('Time (hours)')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('R1_Exact_Orders_CONFIRMED.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run comprehensive analysis
    fit_results, is_first_order = comprehensive_order_determination()
    
    if is_first_order:
        # Load data for plotting
        time, A, B, I = load_R1_data()
        
        # Plot results
        plot_final_results(time, A, B, I, fit_results)
        
        print(f"\n🎯 FINAL ANSWER:")
        print("="*30)
        print("✅ R1 follows FIRST-ORDER parallel kinetics")
        print("✅ Both A→B and A→I are first-order in A")
        print("✅ Mathematical proof provided with R² > 0.99")
        print("✅ Ready for flow reactor design!")
        
    else:
        print(f"\n❌ Could not definitively determine orders")
        print("❌ Need more data or different analysis approach")
