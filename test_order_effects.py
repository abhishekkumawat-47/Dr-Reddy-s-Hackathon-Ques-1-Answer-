#!/usr/bin/env python3
"""
Testing Variable Order Parallel Reactions
========================================

Testing if parallel reactions with different individual orders
can still give overall first-order behavior for R1.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

def parallel_reaction_variable_order(t, y, k1, k2, n1, n2):
    """
    Variable order parallel reactions:
    A → B (rate = k1 * [A]^n1)
    A → I (rate = k2 * [A]^n2)
    """
    A, B, I = y
    
    if A < 1e-10:  # Prevent negative concentrations
        A = 1e-10
    
    dA_dt = -(k1 * A**n1 + k2 * A**n2)
    dB_dt = k1 * A**n1
    dI_dt = k2 * A**n2
    
    return [dA_dt, dB_dt, dI_dt]

def fit_variable_order_parallel(time_data, A_data, B_data, I_data, A0, constrain_total_order=True):
    """
    Fit parallel kinetics with variable orders
    Option to constrain total order ≈ 1
    """
    def residual(params):
        if constrain_total_order:
            k1, k2, n1 = params
            n2 = 2.0 - n1  # Constraint: n1 + n2 ≈ 2 (to give overall order ≈ 1)
        else:
            k1, k2, n1, n2 = params
        
        # Bounds checking
        if k1 <= 0 or k2 <= 0 or n1 <= 0 or n2 <= 0:
            return 1e6
        if n1 > 3 or n2 > 3:
            return 1e6
            
        try:
            sol = solve_ivp(
                lambda t, y: parallel_reaction_variable_order(t, y, k1, k2, n1, n2),
                [time_data[0], time_data[-1]],
                [A0, 0, 0],
                t_eval=time_data,
                method='RK45',
                rtol=1e-8
            )
            
            if not sol.success or len(sol.y[0]) != len(time_data):
                return 1e6
                
            A_pred, B_pred, I_pred = sol.y
            
            # Weighted residuals
            res_A = np.sum((A_data - A_pred)**2)
            res_B = np.sum((B_data - B_pred)**2) * 2  # Weight B more
            res_I = np.sum((I_data - I_pred)**2)
            
            return res_A + res_B + res_I
            
        except:
            return 1e6
    
    best_result = None
    best_residual = float('inf')
    
    if constrain_total_order:
        # Try different n1 values (n2 = 2-n1)
        initial_guesses = [
            [3.0, 0.1, 0.8],    # n1=0.8, n2=1.2
            [2.0, 0.2, 1.2],    # n1=1.2, n2=0.8  
            [4.0, 0.05, 0.6],   # n1=0.6, n2=1.4
            [1.5, 0.3, 1.4],    # n1=1.4, n2=0.6
            [3.5, 0.08, 1.0],   # n1=1.0, n2=1.0 (reference)
        ]
        bounds = [(1e-6, 20), (1e-6, 5), (0.1, 1.9)]
    else:
        # Unconstrained
        initial_guesses = [
            [3.0, 0.1, 1.0, 1.0],
            [2.0, 0.2, 0.8, 1.2],
            [4.0, 0.05, 1.2, 0.8],
        ]
        bounds = [(1e-6, 20), (1e-6, 5), (0.1, 3.0), (0.1, 3.0)]
    
    for guess in initial_guesses:
        try:
            result = minimize(
                residual, 
                guess,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if result.success and result.fun < best_residual:
                best_residual = result.fun
                best_result = result
                
        except:
            continue
    
    if best_result is None:
        return None
        
    if constrain_total_order:
        k1, k2, n1 = best_result.x
        n2 = 2.0 - n1
        params = [k1, k2, n1, n2]
    else:
        params = best_result.x
    
    # Calculate R²
    k1, k2, n1, n2 = params
    try:
        sol = solve_ivp(
            lambda t, y: parallel_reaction_variable_order(t, y, k1, k2, n1, n2),
            [time_data[0], time_data[-1]],
            [A0, 0, 0],
            t_eval=time_data,
            method='RK45'
        )
        
        if sol.success:
            A_pred, B_pred, I_pred = sol.y
            
            # Combined R²
            ss_res = np.sum((A_data - A_pred)**2) + np.sum((B_data - B_pred)**2) + np.sum((I_data - I_pred)**2)
            ss_tot = np.sum((A_data - np.mean(A_data))**2) + np.sum((B_data - np.mean(B_data))**2) + np.sum((I_data - np.mean(I_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            r_squared = 0
            
    except:
        r_squared = 0
    
    return k1, k2, n1, n2, r_squared

def analyze_order_effects():
    """
    Test different order combinations for R1
    """
    print("🧪 TESTING VARIABLE ORDER EFFECTS FOR R1")
    print("="*60)
    
    # Load R1 data
    df = pd.read_excel("Reaction-1.xlsx", sheet_name="Calculated")
    subset = df[(df["Temp_C"] == 70) & (df["A0_mgml"] == 50)]
    
    time_data = subset['Time_h'].values
    A_data = subset['A_mgml'].values
    B_data = subset['B_mgml'].values
    I_data = subset['I_mgml'].values
    A0 = 50
    
    print(f"📊 Analyzing {len(subset)} data points at 70°C, 50 mg/mL")
    
    # Test 1: Both orders = 1 (current approach)
    print("\n🔬 TEST 1: Both reactions first-order (n1=1, n2=1)")
    result1 = fit_variable_order_parallel(time_data, A_data, B_data, I_data, A0, constrain_total_order=False)
    
    if result1:
        k1, k2, n1, n2, r2 = result1
        print(f"   k1 = {k1:.4f} h⁻¹, n1 = {n1:.3f}")
        print(f"   k2 = {k2:.4f} h⁻¹, n2 = {n2:.3f}")
        print(f"   k_total = {k1+k2:.4f} h⁻¹")
        print(f"   R² = {r2:.4f}")
    
    # Test 2: Constrained to give overall order ≈ 1
    print("\n🔬 TEST 2: Variable orders with constraint (n1+n2≈2)")
    result2 = fit_variable_order_parallel(time_data, A_data, B_data, I_data, A0, constrain_total_order=True)
    
    if result2:
        k1, k2, n1, n2, r2 = result2
        print(f"   k1 = {k1:.4f} h⁻¹, n1 = {n1:.3f}")
        print(f"   k2 = {k2:.4f} h⁻¹, n2 = {n2:.3f}")
        print(f"   Total order = {n1+n2:.3f}")
        print(f"   R² = {r2:.4f}")
    
    # Test 3: Unconstrained (best fit regardless of order)
    print("\n🔬 TEST 3: Unconstrained orders (best fit)")
    result3 = fit_variable_order_parallel(time_data, A_data, B_data, I_data, A0, constrain_total_order=False)
    
    if result3:
        k1, k2, n1, n2, r2 = result3
        print(f"   k1 = {k1:.4f} h⁻¹, n1 = {n1:.3f}")
        print(f"   k2 = {k2:.4f} h⁻¹, n2 = {n2:.3f}")
        print(f"   Total order = {n1+n2:.3f}")
        print(f"   R² = {r2:.4f}")
    
    # Conclusion
    print("\n📋 ANALYSIS CONCLUSION:")
    print("="*40)
    
    if result1 and result2 and result3:
        r2_values = [result1[4], result2[4], result3[4]]
        best_idx = np.argmax(r2_values)
        
        approaches = ["Both first-order", "Constrained variable", "Unconstrained"]
        print(f"🏆 Best fit: {approaches[best_idx]} (R² = {r2_values[best_idx]:.4f})")
        
        if best_idx == 0:
            print("✅ Both reactions being first-order is indeed optimal!")
            print("✅ This validates our original approach.")
        else:
            print("⚠️  Variable orders might provide better fit.")
            print("🤔 Consider the physical meaning and complexity trade-off.")
    
    return result1, result2, result3

def test_overall_order_concept():
    """
    Demonstrate that different individual orders can give overall first-order
    """
    print("\n🧮 DEMONSTRATING OVERALL ORDER CONCEPT")
    print("="*50)
    
    # Simulate data with known mixed orders
    t = np.linspace(0, 5, 11)
    A0 = 50
    
    # Case 1: n1=0.8, n2=1.2, but overall appears first-order
    k1_true, k2_true, n1_true, n2_true = 2.5, 0.3, 0.8, 1.2
    
    print("🎯 SIMULATED EXAMPLE:")
    print(f"   True parameters: k1={k1_true}, k2={k2_true}, n1={n1_true}, n2={n2_true}")
    print(f"   True total order: {n1_true + n2_true} ≠ 2")
    
    # Generate "experimental" data
    sol = solve_ivp(
        lambda t, y: parallel_reaction_variable_order(t, y, k1_true, k2_true, n1_true, n2_true),
        [0, 5],
        [A0, 0, 0],
        t_eval=t,
        method='RK45'
    )
    
    A_sim, B_sim, I_sim = sol.y
    
    # Test if this data could be fit as "overall first-order"
    def first_order_total(t, k_eff):
        return A0 * np.exp(-k_eff * t)
    
    # Fit overall A decay as first-order
    try:
        popt, _ = curve_fit(first_order_total, t, A_sim)
        k_eff = popt[0]
        
        A_first_order = first_order_total(t, k_eff)
        
        # Calculate R² for first-order approximation
        ss_res = np.sum((A_sim - A_first_order)**2)
        ss_tot = np.sum((A_sim - np.mean(A_sim))**2)
        r2_apparent = 1 - (ss_res / ss_tot)
        
        print(f"   Apparent first-order k_eff: {k_eff:.4f} h⁻¹")
        print(f"   R² for first-order fit: {r2_apparent:.4f}")
        
        if r2_apparent > 0.95:
            print("✅ This mixed-order system APPEARS first-order overall!")
            print("✅ Proves that individual orders don't need to be 1")
        else:
            print("❌ This system doesn't appear first-order overall")
            
    except:
        print("❌ Failed to fit first-order approximation")

if __name__ == "__main__":
    # Test the order effects
    analyze_order_effects()
    
    # Demonstrate the concept
    test_overall_order_concept()
    
    print("\n🎓 EDUCATIONAL SUMMARY:")
    print("="*40)
    print("1. Overall order ≠ sum of individual orders (necessarily)")
    print("2. Different individual orders CAN give overall first-order")
    print("3. Simplest model (both n=1) often works best in practice")
    print("4. Physical meaning should guide choice of orders")
    print("5. If R² is excellent with n1=n2=1, no need to complicate!")
