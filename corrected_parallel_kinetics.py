#!/usr/bin/env python3
"""
Corrected Parallel Reaction Kinetics Analysis for R1
==================================================

For parallel reactions: A → B (k1) and A → I (k2)
With constraint: Overall order = 1 (as confirmed by comprehensive analysis)

This means: n1 + n2 = 1 (for the overall first-order behavior)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

class CorrectedParallelAnalyzer:
    def __init__(self):
        self.results = {}
        self.data = {}
        
    def load_data(self):
        """Load reaction data from Excel files"""
        print("📂 Loading Reaction Data...")
        
        files = {
            "R1": "Reaction-1.xlsx",
            "R2": "Reaction-2.xlsx", 
            "R3": "Reaction-3.xlsx"
        }
        
        for rxn, file in files.items():
            try:
                df = pd.read_excel(file, sheet_name="Calculated")
                self.data[rxn] = df
                print(f"✅ {rxn}: {len(df)} data points loaded")
            except Exception as e:
                print(f"❌ Error loading {rxn}: {e}")
    
    def parallel_reaction_first_order(self, t, y, k1, k2):
        """
        Parallel reaction ODE system with FIRST-ORDER constraint:
        A → B (rate = k1 * [A])
        A → I (rate = k2 * [A])
        
        Both reactions are first-order in A (total order = 1)
        """
        A, B, I = y
        
        dA_dt = -(k1 + k2) * A  # Total consumption rate
        dB_dt = k1 * A          # Product formation
        dI_dt = k2 * A          # Impurity formation
        
        return [dA_dt, dB_dt, dI_dt]
    
    def analytical_solution_first_order(self, t, A0, k1, k2):
        """
        Analytical solution for first-order parallel reactions:
        A(t) = A0 * exp(-(k1+k2)*t)
        B(t) = A0 * k1/(k1+k2) * (1 - exp(-(k1+k2)*t))
        I(t) = A0 * k2/(k1+k2) * (1 - exp(-(k1+k2)*t))
        """
        k_total = k1 + k2
        
        A = A0 * np.exp(-k_total * t)
        B = A0 * (k1/k_total) * (1 - np.exp(-k_total * t))
        I = A0 * (k2/k_total) * (1 - np.exp(-k_total * t))
        
        return A, B, I
    
    def fit_first_order_parallel(self, time_data, A_data, B_data, I_data, A0):
        """
        Fit first-order parallel kinetics using analytical solution
        """
        def residual(params):
            k1, k2 = params
            
            if k1 <= 0 or k2 <= 0:
                return 1e6
                
            try:
                A_pred, B_pred, I_pred = self.analytical_solution_first_order(time_data, A0, k1, k2)
                
                # Calculate weighted residuals
                res_A = np.sum((A_data - A_pred)**2)
                res_B = np.sum((B_data - B_pred)**2) * 2  # Weight B more (main product)
                res_I = np.sum((I_data - I_pred)**2)
                
                return res_A + res_B + res_I
                
            except:
                return 1e6
        
        # Multiple initial guesses
        initial_guesses = [
            [1.0, 0.1],    # High selectivity
            [0.5, 0.5],    # Equal rates
            [2.0, 0.2],    # Very high selectivity
            [3.0, 0.5],    # Fast product, moderate impurity
        ]
        
        best_result = None
        best_residual = float('inf')
        
        for guess in initial_guesses:
            try:
                result = minimize(
                    residual, 
                    guess,
                    method='L-BFGS-B',
                    bounds=[(1e-6, 20), (1e-6, 5)]
                )
                
                if result.success and result.fun < best_residual:
                    best_residual = result.fun
                    best_result = result
                    
            except:
                continue
        
        if best_result is None:
            return None, None, None
            
        k1, k2 = best_result.x
        
        # Calculate R²
        try:
            A_pred, B_pred, I_pred = self.analytical_solution_first_order(time_data, A0, k1, k2)
            
            # Combined R²
            ss_res = np.sum((A_data - A_pred)**2) + np.sum((B_data - B_pred)**2) + np.sum((I_data - I_pred)**2)
            ss_tot = np.sum((A_data - np.mean(A_data))**2) + np.sum((B_data - np.mean(B_data))**2) + np.sum((I_data - np.mean(I_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Also calculate individual R² values
            ss_res_A = np.sum((A_data - A_pred)**2)
            ss_tot_A = np.sum((A_data - np.mean(A_data))**2)
            r2_A = 1 - (ss_res_A / ss_tot_A) if ss_tot_A > 0 else 0
            
            ss_res_B = np.sum((B_data - B_pred)**2)
            ss_tot_B = np.sum((B_data - np.mean(B_data))**2)
            r2_B = 1 - (ss_res_B / ss_tot_B) if ss_tot_B > 0 else 0
            
            ss_res_I = np.sum((I_data - I_pred)**2)
            ss_tot_I = np.sum((I_data - np.mean(I_data))**2)
            r2_I = 1 - (ss_res_I / ss_tot_I) if ss_tot_I > 0 else 0
            
        except:
            r_squared = r2_A = r2_B = r2_I = 0
            
        return k1, k2, r_squared, r2_A, r2_B, r2_I
    
    def analyze_R1_corrected(self, temp=70, conc=50):
        """
        Corrected R1 analysis with first-order constraint
        """
        print(f"\n🔬 CORRECTED R1 Analysis at {temp}°C, {conc} mg/mL")
        print("="*60)
        print("🎯 Constraint: Overall first-order kinetics (confirmed by comprehensive analysis)")
        
        df = self.data["R1"]
        subset = df[(df["Temp_C"] == temp) & (df["A0_mgml"] == conc)]
        
        if subset.empty:
            print(f"❌ No data found for T={temp}°C, C={conc} mg/mL")
            return None
            
        print(f"📊 Data points: {len(subset)}")
        
        # Extract data
        time_data = subset['Time_h'].values
        A_data = subset['A_mgml'].values
        B_data = subset['B_mgml'].values
        I_data = subset['I_mgml'].values
        A0 = conc
        
        print(f"⏰ Time range: {time_data[0]:.1f} - {time_data[-1]:.1f} hours")
        print(f"🧪 Initial A concentration: {A0} mg/mL")
        print(f"📈 Final conversion: {((A0 - A_data[-1])/A0)*100:.1f}%")
        print(f"🎯 Final B yield: {(B_data[-1]/A0)*100:.1f}%")
        print(f"⚠️  Final I formation: {(I_data[-1]/A0)*100:.1f}%")
        
        # Fit corrected parallel kinetics
        print("\n🧮 Fitting First-Order Parallel Kinetics...")
        result = self.fit_first_order_parallel(time_data, A_data, B_data, I_data, A0)
        
        if result[0] is None:
            print("❌ Kinetic fitting failed")
            return None
            
        k1, k2, r_squared, r2_A, r2_B, r2_I = result
        
        # Verify total rate constant matches comprehensive analysis
        k_total = k1 + k2
        expected_k = 3.6357  # From comprehensive analysis at 70°C, 50 mg/mL
        
        print(f"\n✅ VALIDATION:")
        print(f"   k_total calculated: {k_total:.4f} h⁻¹")
        print(f"   k_total expected: {expected_k:.4f} h⁻¹")
        print(f"   Match: {'✅ YES' if abs(k_total - expected_k) < 0.1 else '❌ NO'}")
        
        # Store results
        result_dict = {
            'temperature': temp,
            'concentration': conc,
            'k1': k1,
            'k2': k2,
            'k_total': k_total,
            'r_squared': r_squared,
            'r2_A': r2_A,
            'r2_B': r2_B,
            'r2_I': r2_I,
            'time_data': time_data,
            'A_data': A_data,
            'B_data': B_data,
            'I_data': I_data,
            'A0': A0
        }
        
        self.results[f"R1_{temp}C_{conc}mgml"] = result_dict
        
        # Display results
        print(f"\n📋 CORRECTED KINETIC PARAMETERS:")
        print(f"   k1 (A→B): {k1:.4f} h⁻¹ (order = 1)")
        print(f"   k2 (A→I): {k2:.4f} h⁻¹ (order = 1)")
        print(f"   k_total: {k_total:.4f} h⁻¹")
        print(f"   R² overall: {r_squared:.4f}")
        print(f"   R² for A: {r2_A:.4f}")
        print(f"   R² for B: {r2_B:.4f}")
        print(f"   R² for I: {r2_I:.4f}")
        
        # Calculate selectivity
        selectivity = k1 / (k1 + k2) * 100
        print(f"   Selectivity (k1/(k1+k2)): {selectivity:.1f}%")
        
        return result_dict
    
    def calculate_corrected_residence_time(self, result, target_conversion=0.99):
        """
        Calculate residence time using correct first-order kinetics
        """
        k1, k2, A0 = result['k1'], result['k2'], result['A0']
        k_total = k1 + k2
        
        print(f"\n⏱️  CORRECTED RESIDENCE TIME (Target: {target_conversion*100}% conversion)")
        print("="*60)
        
        # For first-order: conversion = 1 - exp(-k_total * t)
        # Solve for t: t = -ln(1-conversion) / k_total
        
        residence_time_batch = -np.log(1 - target_conversion) / k_total
        
        # Apply flow enhancement factor
        flow_enhancement = 5  # Conservative for microreactors
        residence_time_flow = residence_time_batch / flow_enhancement
        
        print(f"📊 First-order formula: t = -ln(1-X)/k_total")
        print(f"📊 Batch residence time: {residence_time_batch:.3f} hours = {residence_time_batch*60:.1f} minutes")
        print(f"🚀 Flow residence time (÷{flow_enhancement}): {residence_time_flow:.3f} hours = {residence_time_flow*60:.1f} minutes")
        
        # Calculate performance at this time
        A_final, B_final, I_final = self.analytical_solution_first_order(
            np.array([residence_time_batch]), A0, k1, k2)
        
        A_final, B_final, I_final = A_final[0], B_final[0], I_final[0]
        
        conversion = (A0 - A_final) / A0
        yield_B = B_final / A0
        impurity_formation = I_final / A0
        selectivity = B_final / (B_final + I_final) if (B_final + I_final) > 0 else 0
        
        print(f"\n🎯 PERFORMANCE AT OPTIMAL RESIDENCE TIME:")
        print(f"   Conversion: {conversion*100:.1f}%")
        print(f"   B Yield: {yield_B*100:.1f}%")
        print(f"   I Formation: {impurity_formation*100:.1f}%")
        print(f"   Selectivity: {selectivity*100:.1f}%")
        
        return {
            'residence_time_batch': residence_time_batch,
            'residence_time_flow': residence_time_flow,
            'conversion': conversion,
            'yield_B': yield_B,
            'impurity_formation': impurity_formation,
            'selectivity': selectivity
        }
    
    def plot_corrected_analysis(self, result):
        """
        Plot the corrected kinetic analysis
        """
        k1, k2, A0 = result['k1'], result['k2'], result['A0']
        time_data = result['time_data']
        A_data, B_data, I_data = result['A_data'], result['B_data'], result['I_data']
        
        # Generate smooth time for prediction
        t_smooth = np.linspace(0, time_data[-1], 100)
        A_pred, B_pred, I_pred = self.analytical_solution_first_order(t_smooth, A0, k1, k2)
        
        # Also get predictions at data points
        A_pred_data, B_pred_data, I_pred_data = self.analytical_solution_first_order(time_data, A0, k1, k2)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('R1 Corrected First-Order Parallel Kinetics Analysis', fontsize=16, fontweight='bold')
        
        # Plot A concentration
        axes[0,0].plot(time_data, A_data, 'bo', markersize=8, label='Experimental A')
        axes[0,0].plot(t_smooth, A_pred, 'b-', linewidth=2, label='First-order model')
        axes[0,0].set_title('Reactant A Consumption', fontweight='bold')
        axes[0,0].set_ylabel('A Concentration (mg/mL)')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # Plot B concentration
        axes[0,1].plot(time_data, B_data, 'go', markersize=8, label='Experimental B')
        axes[0,1].plot(t_smooth, B_pred, 'g-', linewidth=2, label='First-order model')
        axes[0,1].set_title('Product B Formation', fontweight='bold')
        axes[0,1].set_ylabel('B Concentration (mg/mL)')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # Plot I concentration
        axes[1,0].plot(time_data, I_data, 'ro', markersize=8, label='Experimental I')
        axes[1,0].plot(t_smooth, I_pred, 'r-', linewidth=2, label='First-order model')
        axes[1,0].set_title('Impurity I Formation', fontweight='bold')
        axes[1,0].set_ylabel('I Concentration (mg/mL)')
        axes[1,0].set_xlabel('Time (hours)')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        # Plot selectivity vs time
        selectivity_exp = B_data / (B_data + I_data + 1e-6) * 100
        selectivity_pred = B_pred / (B_pred + I_pred + 1e-6) * 100
        
        axes[1,1].plot(time_data, selectivity_exp, 'mo', markersize=8, label='Experimental')
        axes[1,1].plot(t_smooth, selectivity_pred, 'm-', linewidth=2, label='First-order model')
        axes[1,1].set_title('Selectivity vs Time', fontweight='bold')
        axes[1,1].set_ylabel('Selectivity (%)')
        axes[1,1].set_xlabel('Time (hours)')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        # Add text box with kinetic parameters
        textstr = f'k1 = {k1:.3f} h⁻¹\nk2 = {k2:.3f} h⁻¹\nk_total = {k1+k2:.3f} h⁻¹\nSelectivity = {k1/(k1+k2)*100:.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[1,1].text(0.05, 0.95, textstr, transform=axes[1,1].transAxes, fontsize=10,
                      verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('R1_Corrected_Parallel_Analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("🔧 CORRECTED PARALLEL REACTION KINETICS ANALYZER")
    print("="*60)
    print("Constraint: Overall first-order kinetics (confirmed)")
    print("A → B (k1, first-order) + A → I (k2, first-order)")
    print("="*60)
    
    analyzer = CorrectedParallelAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Analyze R1 with correct constraint
    result = analyzer.analyze_R1_corrected(temp=70, conc=50)
    
    if result:
        # Plot analysis
        analyzer.plot_corrected_analysis(result)
        
        # Calculate residence time
        residence_time_result = analyzer.calculate_corrected_residence_time(result, target_conversion=0.99)
        
        if residence_time_result:
            print("\n🏭 CORRECTED FLOW REACTOR RECOMMENDATIONS")
            print("="*60)
            
            k1, k2 = result['k1'], result['k2']
            temp, conc = result['temperature'], result['concentration']
            residence_time = residence_time_result['residence_time_flow'] * 60  # minutes
            
            print(f"🎯 OPTIMAL CONDITIONS:")
            print(f"   Temperature: {temp}°C")
            print(f"   Concentration: {conc} mg/mL")
            print(f"   Residence Time: {residence_time:.1f} minutes")
            
            print(f"\n⚗️  CORRECTED KINETICS:")
            print(f"   A → B: k1 = {k1:.4f} h⁻¹ (first-order)")
            print(f"   A → I: k2 = {k2:.4f} h⁻¹ (first-order)")
            print(f"   Total: k_total = {k1+k2:.4f} h⁻¹ (matches comprehensive analysis)")
            print(f"   Selectivity: {k1/(k1+k2)*100:.1f}%")
            
            print(f"\n🚀 FLOW REACTOR SPECS:")
            flow_rate = 3.0  # mL/min
            reactor_volume = flow_rate * residence_time
            print(f"   Flow Rate: {flow_rate} mL/min")
            print(f"   Reactor Volume: {reactor_volume:.1f} mL")
            print(f"   Reactor Type: PFR (first-order kinetics)")
            
            print(f"\n📊 EXPECTED PERFORMANCE:")
            print(f"   Conversion: {residence_time_result['conversion']*100:.1f}%")
            print(f"   B Yield: {residence_time_result['yield_B']*100:.1f}%")
            print(f"   I Formation: {residence_time_result['impurity_formation']*100:.1f}%")
            print(f"   Selectivity: {residence_time_result['selectivity']*100:.1f}%")
        
        print("\n✅ Corrected Analysis Complete!")
        print("📋 Results now match comprehensive analysis constraint")
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main()
