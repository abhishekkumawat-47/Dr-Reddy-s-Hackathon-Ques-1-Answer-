#!/usr/bin/env python3
"""
Exact Order Determination for Parallel Reactions
==============================================

Mathematically determines exact individual reaction orders for:
A → B (rate = k1 * [A]^n1)
A → I (rate = k2 * [A]^n2)

Using differential rate analysis and integral methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class ExactOrderDetermination:
    def __init__(self):
        self.results = {}
        
    def load_R1_data(self, temp=70, conc=50):
        """Load R1 data at optimal conditions"""
        print(f"📂 Loading R1 data at {temp}°C, {conc} mg/mL")
        
        df = pd.read_excel("Reaction-1.xlsx", sheet_name="Calculated")
        subset = df[(df["Temp_C"] == temp) & (df["A0_mgml"] == conc)]
        
        if subset.empty:
            raise ValueError(f"No data found for T={temp}°C, C={conc} mg/mL")
        
        # Extract and clean data
        time = subset['Time_h'].values
        A = subset['A_mgml'].values
        B = subset['B_mgml'].values
        I = subset['I_mgml'].values
        
        return time, A, B, I
    
    def calculate_rates(self, time, concentration):
        """
        Calculate instantaneous rates using numerical differentiation
        with smoothing to reduce noise
        """
        # Smooth the data first
        if len(concentration) > 5:
            smoothed = savgol_filter(concentration, window_length=5, polyorder=2)
        else:
            smoothed = concentration
        
        # Calculate derivatives
        rates = np.gradient(smoothed, time)
        
        return rates, smoothed
    
    def determine_order_differential_method(self, time, A, B, I):
        """
        Method 1: Differential rate analysis
        For A → B: d[B]/dt = k1 * [A]^n1
        For A → I: d[I]/dt = k2 * [A]^n2
        
        Taking logarithms: ln(rate) = ln(k) + n*ln([A])
        Slope of ln(rate) vs ln([A]) gives order n
        """
        print("\n🔬 METHOD 1: Differential Rate Analysis")
        print("="*50)
        
        # Calculate rates
        dB_dt, B_smooth = self.calculate_rates(time, B)
        dI_dt, I_smooth = self.calculate_rates(time, I)
        dA_dt, A_smooth = self.calculate_rates(time, A)
        
        # Remove negative rates and zero concentrations
        valid_B = (dB_dt > 1e-6) & (A_smooth > 1e-3)
        valid_I = (dI_dt > 1e-6) & (A_smooth > 1e-3)
        
        results = {}
        
        if np.sum(valid_B) > 3:
            # For B formation: ln(dB/dt) vs ln([A])
            ln_rate_B = np.log(dB_dt[valid_B])
            ln_A_B = np.log(A_smooth[valid_B])
            
            # Linear regression
            coeffs_B = np.polyfit(ln_A_B, ln_rate_B, 1)
            n1_diff = coeffs_B[0]
            ln_k1_diff = coeffs_B[1]
            k1_diff = np.exp(ln_k1_diff)
            
            # Calculate R²
            y_pred = np.polyval(coeffs_B, ln_A_B)
            ss_res = np.sum((ln_rate_B - y_pred)**2)
            ss_tot = np.sum((ln_rate_B - np.mean(ln_rate_B))**2)
            r2_B = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            results['B'] = {
                'order': n1_diff,
                'k': k1_diff,
                'r_squared': r2_B,
                'method': 'differential'
            }
            
            print(f"   A → B: n1 = {n1_diff:.3f}, k1 = {k1_diff:.4f} h⁻¹, R² = {r2_B:.4f}")
        
        if np.sum(valid_I) > 3:
            # For I formation: ln(dI/dt) vs ln([A])
            ln_rate_I = np.log(dI_dt[valid_I])
            ln_A_I = np.log(A_smooth[valid_I])
            
            # Linear regression
            coeffs_I = np.polyfit(ln_A_I, ln_rate_I, 1)
            n2_diff = coeffs_I[0]
            ln_k2_diff = coeffs_I[1]
            k2_diff = np.exp(ln_k2_diff)
            
            # Calculate R²
            y_pred = np.polyval(coeffs_I, ln_A_I)
            ss_res = np.sum((ln_rate_I - y_pred)**2)
            ss_tot = np.sum((ln_rate_I - np.mean(ln_rate_I))**2)
            r2_I = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            results['I'] = {
                'order': n2_diff,
                'k': k2_diff,
                'r_squared': r2_I,
                'method': 'differential'
            }
            
            print(f"   A → I: n2 = {n2_diff:.3f}, k2 = {k2_diff:.4f} h⁻¹, R² = {r2_I:.4f}")
        
        return results
    
    def determine_order_integral_method(self, time, A, B, I, A0):
        """
        Method 2: Integral analysis
        For different orders, integrated forms are:
        - 0th order: [A] = [A0] - kt
        - 1st order: ln([A0]/[A]) = kt  
        - 2nd order: 1/[A] - 1/[A0] = kt
        - nth order: ([A0]^(1-n) - [A]^(1-n))/((1-n)*[A0]^(1-n)) = kt/[A0]
        
        Best linear fit determines the order
        """
        print("\n🔬 METHOD 2: Integral Analysis")
        print("="*45)
        
        # Test different orders for A consumption
        orders_to_test = np.arange(0.1, 3.1, 0.1)
        best_order = None
        best_r2 = -1
        best_k = None
        
        # Remove zero and negative concentrations
        valid_idx = (A > 1e-6) & (time >= 0)
        time_clean = time[valid_idx]
        A_clean = A[valid_idx]
        
        if len(A_clean) < 4:
            print("❌ Insufficient data points for integral analysis")
            return {}
        
        print("   Testing orders from 0.1 to 3.0...")
        
        for n in orders_to_test:
            try:
                if abs(n - 1.0) < 0.05:  # First order
                    y = np.log(A0 / A_clean)
                elif abs(n) < 0.05:  # Zero order
                    y = A0 - A_clean
                elif abs(n - 2.0) < 0.05:  # Second order
                    y = 1/A_clean - 1/A0
                else:  # General nth order
                    if n != 1:
                        y = (A0**(1-n) - A_clean**(1-n)) / ((1-n) * A0**(1-n))
                    else:
                        continue
                
                # Linear regression: y = k*t
                if len(y) == len(time_clean) and np.all(np.isfinite(y)):
                    coeffs = np.polyfit(time_clean, y, 1)
                    k_apparent = coeffs[0]
                    
                    if k_apparent > 0:  # Physical constraint
                        y_pred = np.polyval(coeffs, time_clean)
                        ss_res = np.sum((y - y_pred)**2)
                        ss_tot = np.sum((y - np.mean(y))**2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_order = n
                            best_k = k_apparent
            except:
                continue
        
        print(f"   🎯 Best overall order for A consumption: {best_order:.2f}")
        print(f"   🎯 Rate constant: {best_k:.4f} h⁻¹")
        print(f"   🎯 R²: {best_r2:.4f}")
        
        return {
            'overall': {
                'order': best_order,
                'k_total': best_k,
                'r_squared': best_r2,
                'method': 'integral'
            }
        }
    
    def determine_exact_parallel_orders(self, time, A, B, I, A0):
        """
        Method 3: Exact parallel order determination
        Using the constraint that total consumption matches observed A decay
        """
        print("\n🔬 METHOD 3: Exact Parallel Order Determination")
        print("="*55)
        
        # First, get the exact overall order from integral method
        integral_result = self.determine_order_integral_method(time, A, B, I, A0)
        
        if 'overall' not in integral_result:
            print("❌ Could not determine overall order")
            return {}
        
        n_total = integral_result['overall']['order']
        k_total = integral_result['overall']['k_total']
        
        print(f"   Overall order constraint: {n_total:.3f}")
        print(f"   Overall rate constant: {k_total:.4f} h⁻¹")
        
        # Now determine individual orders using the fact that:
        # d[A]/dt = -(k1*[A]^n1 + k2*[A]^n2)
        # d[B]/dt = k1*[A]^n1  
        # d[I]/dt = k2*[A]^n2
        
        # Calculate actual rates
        dB_dt, B_smooth = self.calculate_rates(time, B)
        dI_dt, I_smooth = self.calculate_rates(time, I)
        
        # For each time point, we have:
        # dB_dt = k1 * A^n1
        # dI_dt = k2 * A^n2
        
        # Taking ratios eliminates time dependence:
        # dB_dt / dI_dt = (k1/k2) * A^(n1-n2)
        
        valid_idx = (dB_dt > 1e-6) & (dI_dt > 1e-6) & (A > 1e-3)
        
        if np.sum(valid_idx) < 4:
            print("❌ Insufficient valid data points for parallel analysis")
            return integral_result
        
        B_rates = dB_dt[valid_idx]
        I_rates = dI_dt[valid_idx]
        A_vals = A[valid_idx]
        
        # Method 3a: Assume n1 = 1 (common), solve for n2
        print("\n   🧮 Testing n1 = 1 assumption:")
        
        def test_n1_equals_1():
            # If n1 = 1: dB_dt = k1 * A
            # So k1 = dB_dt / A for each point
            k1_values = B_rates / A_vals
            k1_mean = np.mean(k1_values)
            k1_std = np.std(k1_values)
            cv_k1 = k1_std / k1_mean if k1_mean > 0 else float('inf')
            
            print(f"      k1 mean: {k1_mean:.4f} ± {k1_std:.4f} h⁻¹ (CV: {cv_k1:.3f})")
            
            # If n2 = 1: dI_dt = k2 * A  
            k2_values = I_rates / A_vals
            k2_mean = np.mean(k2_values)
            k2_std = np.std(k2_values)
            cv_k2 = k2_std / k2_mean if k2_mean > 0 else float('inf')
            
            print(f"      k2 mean: {k2_mean:.4f} ± {k2_std:.4f} h⁻¹ (CV: {cv_k2:.3f})")
            print(f"      k_total: {k1_mean + k2_mean:.4f} h⁻¹ (expected: {k_total:.4f})")
            
            # Check consistency
            total_match = abs((k1_mean + k2_mean) - k_total) / k_total
            print(f"      Total rate match: {(1-total_match)*100:.1f}%")
            
            consistency_score = (1 - cv_k1) * (1 - cv_k2) * (1 - total_match)
            print(f"      Consistency score: {consistency_score:.3f}")
            
            return {
                'n1': 1.0,
                'n2': 1.0,
                'k1': k1_mean,
                'k2': k2_mean,
                'k1_cv': cv_k1,
                'k2_cv': cv_k2,
                'total_match': 1 - total_match,
                'consistency': consistency_score
            }
        
        # Method 3b: Use optimization to find best n1, n2
        print("\n   🧮 Optimization approach:")
        
        def optimize_orders():
            def objective(params):
                n1, n2 = params
                
                # Calculate k1 and k2 for each point
                k1_values = B_rates / (A_vals ** n1)
                k2_values = I_rates / (A_vals ** n2)
                
                # Minimize coefficient of variation
                cv1 = np.std(k1_values) / np.mean(k1_values) if np.mean(k1_values) > 0 else 10
                cv2 = np.std(k2_values) / np.mean(k2_values) if np.mean(k2_values) > 0 else 10
                
                # Also minimize deviation from total rate
                k_total_calc = np.mean(k1_values) + np.mean(k2_values)
                total_error = abs(k_total_calc - k_total) / k_total
                
                return cv1 + cv2 + total_error * 5  # Weight total error more
            
            # Bounds: orders between 0.1 and 3.0
            from scipy.optimize import minimize
            
            best_result = None
            best_objective = float('inf')
            
            # Try multiple starting points
            for n1_start in [0.5, 1.0, 1.5]:
                for n2_start in [0.5, 1.0, 1.5]:
                    try:
                        result = minimize(
                            objective,
                            [n1_start, n2_start],
                            bounds=[(0.1, 3.0), (0.1, 3.0)],
                            method='L-BFGS-B'
                        )
                        
                        if result.success and result.fun < best_objective:
                            best_objective = result.fun
                            best_result = result
                    except:
                        continue
            
            if best_result:
                n1_opt, n2_opt = best_result.x
                
                # Calculate final parameters
                k1_values = B_rates / (A_vals ** n1_opt)
                k2_values = I_rates / (A_vals ** n2_opt)
                
                k1_final = np.mean(k1_values)
                k2_final = np.mean(k2_values)
                k1_cv = np.std(k1_values) / k1_final if k1_final > 0 else float('inf')
                k2_cv = np.std(k2_values) / k2_final if k2_final > 0 else float('inf')
                
                print(f"      Optimal n1: {n1_opt:.3f}")
                print(f"      Optimal n2: {n2_opt:.3f}")
                print(f"      k1: {k1_final:.4f} h⁻¹ (CV: {k1_cv:.3f})")
                print(f"      k2: {k2_final:.4f} h⁻¹ (CV: {k2_cv:.3f})")
                print(f"      k_total: {k1_final + k2_final:.4f} h⁻¹")
                
                return {
                    'n1': n1_opt,
                    'n2': n2_opt,
                    'k1': k1_final,
                    'k2': k2_final,
                    'k1_cv': k1_cv,
                    'k2_cv': k2_cv,
                    'objective': best_objective
                }
            
            return None
        
        # Run both methods
        method_3a = test_n1_equals_1()
        method_3b = optimize_orders()
        
        # Compare and choose best
        print(f"\n   📊 COMPARISON:")
        print(f"      Both first-order: consistency = {method_3a['consistency']:.3f}")
        if method_3b:
            print(f"      Optimized orders: objective = {method_3b['objective']:.3f}")
            
            # Choose best method
            if method_3a['consistency'] > 0.8 and (not method_3b or method_3a['consistency'] > 0.9):
                chosen = method_3a
                print(f"   ✅ SELECTED: Both reactions are first-order")
            elif method_3b:
                chosen = method_3b
                print(f"   ✅ SELECTED: Optimized orders (n1={chosen['n1']:.2f}, n2={chosen['n2']:.2f})")
            else:
                chosen = method_3a
                print(f"   ✅ SELECTED: Default to first-order (optimization failed)")
        else:
            chosen = method_3a
            print(f"   ✅ SELECTED: First-order (optimization failed)")
        
        # Add to results
        integral_result['parallel'] = {
            'n1': chosen['n1'],
            'n2': chosen['n2'], 
            'k1': chosen['k1'],
            'k2': chosen['k2'],
            'method': 'exact_parallel',
            'validation': chosen
        }
        
        return integral_result
    
    def validate_results(self, time, A, B, I, A0, results):
        """
        Validate the determined orders by simulation
        """
        print(f"\n✅ VALIDATION OF DETERMINED ORDERS")
        print("="*45)
        
        if 'parallel' not in results:
            print("❌ No parallel results to validate")
            return
        
        n1 = results['parallel']['n1']
        n2 = results['parallel']['n2']
        k1 = results['parallel']['k1']
        k2 = results['parallel']['k2']
        
        print(f"   Validating: n1={n1:.3f}, n2={n2:.3f}, k1={k1:.4f}, k2={k2:.4f}")
        
        # Simulate using determined parameters
        def parallel_ode(t, y):
            A_val, B_val, I_val = y
            if A_val < 1e-10:
                A_val = 1e-10
            
            dA_dt = -(k1 * A_val**n1 + k2 * A_val**n2)
            dB_dt = k1 * A_val**n1
            dI_dt = k2 * A_val**n2
            
            return [dA_dt, dB_dt, dI_dt]
        
        try:
            sol = solve_ivp(
                parallel_ode,
                [time[0], time[-1]],
                [A0, 0, 0],
                t_eval=time,
                method='RK45',
                rtol=1e-8
            )
            
            if sol.success:
                A_pred, B_pred, I_pred = sol.y
                
                # Calculate R² for each component
                def calc_r2(actual, predicted):
                    ss_res = np.sum((actual - predicted)**2)
                    ss_tot = np.sum((actual - np.mean(actual))**2)
                    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                r2_A = calc_r2(A, A_pred)
                r2_B = calc_r2(B, B_pred)
                r2_I = calc_r2(I, I_pred)
                r2_overall = (r2_A + r2_B + r2_I) / 3
                
                print(f"   R² for A: {r2_A:.4f}")
                print(f"   R² for B: {r2_B:.4f}")
                print(f"   R² for I: {r2_I:.4f}")
                print(f"   R² overall: {r2_overall:.4f}")
                
                # Final values comparison
                print(f"\n   Final Values Comparison:")
                print(f"      A: experimental = {A[-1]:.2f}, predicted = {A_pred[-1]:.2f}")
                print(f"      B: experimental = {B[-1]:.2f}, predicted = {B_pred[-1]:.2f}")
                print(f"      I: experimental = {I[-1]:.2f}, predicted = {I_pred[-1]:.2f}")
                
                # Add validation to results
                results['validation'] = {
                    'r2_A': r2_A,
                    'r2_B': r2_B,
                    'r2_I': r2_I,
                    'r2_overall': r2_overall,
                    'A_pred': A_pred,
                    'B_pred': B_pred,
                    'I_pred': I_pred
                }
                
                if r2_overall > 0.95:
                    print(f"   ✅ EXCELLENT fit - orders are correct!")
                elif r2_overall > 0.90:
                    print(f"   ✅ GOOD fit - orders are acceptable")
                else:
                    print(f"   ⚠️  MODERATE fit - consider alternative orders")
                    
            else:
                print("   ❌ Simulation failed")
                
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
    
    def plot_results(self, time, A, B, I, results):
        """
        Plot experimental data vs model predictions
        """
        if 'validation' not in results:
            print("❌ No validation data to plot")
            return
        
        A_pred = results['validation']['A_pred']
        B_pred = results['validation']['B_pred']
        I_pred = results['validation']['I_pred']
        
        n1 = results['parallel']['n1']
        n2 = results['parallel']['n2']
        k1 = results['parallel']['k1']
        k2 = results['parallel']['k2']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Exact Order Determination Results\nn1={n1:.3f}, n2={n2:.3f}, k1={k1:.4f}, k2={k2:.4f}', 
                     fontsize=14, fontweight='bold')
        
        # A concentration
        axes[0,0].plot(time, A, 'bo', markersize=8, label='Experimental')
        axes[0,0].plot(time, A_pred, 'b-', linewidth=2, label=f'Model (R²={results["validation"]["r2_A"]:.3f})')
        axes[0,0].set_title('Reactant A Consumption')
        axes[0,0].set_ylabel('A Concentration (mg/mL)')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # B concentration
        axes[0,1].plot(time, B, 'go', markersize=8, label='Experimental')
        axes[0,1].plot(time, B_pred, 'g-', linewidth=2, label=f'Model (R²={results["validation"]["r2_B"]:.3f})')
        axes[0,1].set_title(f'Product B Formation (order={n1:.2f})')
        axes[0,1].set_ylabel('B Concentration (mg/mL)')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # I concentration
        axes[1,0].plot(time, I, 'ro', markersize=8, label='Experimental')
        axes[1,0].plot(time, I_pred, 'r-', linewidth=2, label=f'Model (R²={results["validation"]["r2_I"]:.3f})')
        axes[1,0].set_title(f'Impurity I Formation (order={n2:.2f})')
        axes[1,0].set_ylabel('I Concentration (mg/mL)')
        axes[1,0].set_xlabel('Time (hours)')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        # Selectivity
        sel_exp = B / (B + I) * 100
        sel_pred = B_pred / (B_pred + I_pred) * 100
        
        axes[1,1].plot(time, sel_exp, 'mo', markersize=8, label='Experimental')
        axes[1,1].plot(time, sel_pred, 'm-', linewidth=2, label='Model')
        axes[1,1].set_title('Selectivity vs Time')
        axes[1,1].set_ylabel('Selectivity (%)')
        axes[1,1].set_xlabel('Time (hours)')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('R1_Exact_Order_Determination.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("🔬 EXACT ORDER DETERMINATION FOR R1 PARALLEL REACTIONS")
    print("="*65)
    print("Mathematical determination of individual reaction orders")
    print("A → B (k1, order n1) + A → I (k2, order n2)")
    print("="*65)
    
    analyzer = ExactOrderDetermination()
    
    try:
        # Load data
        time, A, B, I = analyzer.load_R1_data(temp=70, conc=50)
        A0 = 50
        
        print(f"📊 Loaded {len(time)} data points")
        print(f"⏰ Time range: {time[0]:.1f} - {time[-1]:.1f} hours")
        print(f"📈 Final conversion: {((A0 - A[-1])/A0)*100:.1f}%")
        print(f"🎯 Final B yield: {(B[-1]/A0)*100:.1f}%")
        print(f"⚠️  Final I formation: {(I[-1]/A0)*100:.1f}%")
        
        # Method 1: Differential analysis
        diff_results = analyzer.determine_order_differential_method(time, A, B, I)
        
        # Method 2 & 3: Integral and exact parallel analysis
        exact_results = analyzer.determine_exact_parallel_orders(time, A, B, I, A0)
        
        # Validate results
        analyzer.validate_results(time, A, B, I, A0, exact_results)
        
        # Plot results
        analyzer.plot_results(time, A, B, I, exact_results)
        
        # Final summary
        print(f"\n🏆 FINAL EXACT ORDER DETERMINATION")
        print("="*50)
        
        if 'parallel' in exact_results:
            n1 = exact_results['parallel']['n1']
            n2 = exact_results['parallel']['n2']
            k1 = exact_results['parallel']['k1']
            k2 = exact_results['parallel']['k2']
            
            print(f"✅ A → B: order = {n1:.3f}, k1 = {k1:.4f} h⁻¹")
            print(f"✅ A → I: order = {n2:.3f}, k2 = {k2:.4f} h⁻¹")
            print(f"✅ Total: k_total = {k1+k2:.4f} h⁻¹")
            print(f"✅ Selectivity: {k1/(k1+k2)*100:.1f}%")
            
            if 'validation' in exact_results:
                r2 = exact_results['validation']['r2_overall']
                print(f"✅ Model fit: R² = {r2:.4f}")
                
            # Calculate residence time with exact orders
            conversion = 0.99
            if abs(n1 - 1.0) < 0.01 and abs(n2 - 1.0) < 0.01:
                # Both first-order: use analytical solution
                residence_time = -np.log(1 - conversion) / (k1 + k2)
                print(f"✅ Residence time (99% conv): {residence_time*60:.1f} minutes")
            else:
                print(f"✅ Residence time: Requires numerical solution for these orders")
                
        else:
            print("❌ Could not determine exact orders")
        
        # Store results
        analyzer.results = exact_results
        
        print(f"\n📋 Analysis complete! Check 'R1_Exact_Order_Determination.png'")
        
        return exact_results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

if __name__ == "__main__":
    results = main()
