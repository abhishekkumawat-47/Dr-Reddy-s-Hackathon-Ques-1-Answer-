import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def analyze_all_reactions():
    """
    Comprehensive analysis of all reactions across all conditions
    """
    
    def general_order_model(t, A0, k, n):
        """General nth order kinetics"""
        if abs(n - 1.0) < 1e-6:  # First order case
            return A0 * np.exp(-k * t)
        else:
            term = 1 + (n - 1) * k * (A0 ** (n - 1)) * t
            return A0 * (term ** (-1 / (n - 1)))
    
    files = {
        "R1": "Reaction-1.xlsx",
        "R2": "Reaction-2.xlsx", 
        "R3": "Reaction-3.xlsx"
    }
    
    all_results = []
    
    for rxn, file in files.items():
        print(f"\n{'='*60}")
        print(f"🧪 ANALYZING {rxn} - ALL CONDITIONS")
        print(f"{'='*60}")
        
        # Load data
        df = pd.read_excel(file, sheet_name="Calculated")
        
        # Get all unique conditions
        conditions = []
        for _, row in df.iterrows():
            condition = (row['Temp_C'], row['A0_mgml'])
            if condition not in conditions:
                conditions.append(condition)
        
        print(f"📊 Total conditions found: {len(conditions)}")
        
        reaction_orders = []
        rate_constants = []
        r_squared_values = []
        
        for temp, conc in sorted(conditions):
            subset = df[(df["Temp_C"] == temp) & (df["A0_mgml"] == conc)]
            
            if len(subset) < 3:
                continue
                
            time_data = subset['Time_h'].values
            conc_data = subset['A_mgml'].values
            
            # Remove invalid data
            valid_mask = (conc_data > 0) & (time_data >= 0)
            time_data = time_data[valid_mask]
            conc_data = conc_data[valid_mask]
            
            if len(time_data) < 3:
                continue
                
            A0_initial = conc_data[0]
            
            # Try to fit general order model
            try:
                popt, pcov = curve_fit(
                    general_order_model, 
                    time_data, 
                    conc_data, 
                    p0=[A0_initial, 0.1, 1.0],
                    bounds=([A0_initial*0.9, 0.001, 0.1], [A0_initial*1.1, 10.0, 3.0]),
                    maxfev=10000
                )
                
                A0_fit, k_fit, n_fit = popt
                
                # Calculate R²
                y_pred = general_order_model(time_data, *popt)
                ss_res = np.sum((conc_data - y_pred) ** 2)
                ss_tot = np.sum((conc_data - np.mean(conc_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Calculate standard error
                param_errors = np.sqrt(np.diag(pcov))
                n_error = param_errors[2]
                
                reaction_orders.append(n_fit)
                rate_constants.append(k_fit)
                r_squared_values.append(r_squared)
                
                print(f"T={temp}°C, C={conc}mg/mL: Order={n_fit:.4f}±{n_error:.4f}, k={k_fit:.4f}, R²={r_squared:.6f}")
                
                all_results.append({
                    'Reaction': rxn,
                    'Temperature_C': temp,
                    'Concentration_mgmL': conc,
                    'Reaction_Order': n_fit,
                    'Order_Error': n_error,
                    'Rate_Constant': k_fit,
                    'R_Squared': r_squared,
                    'Data_Points': len(time_data)
                })
                
            except Exception as e:
                print(f"T={temp}°C, C={conc}mg/mL: ❌ Fitting failed")
        
        # Summary statistics for this reaction
        if reaction_orders:
            print(f"\n📈 {rxn} SUMMARY:")
            print(f"  Average Order: {np.mean(reaction_orders):.4f} ± {np.std(reaction_orders):.4f}")
            print(f"  Order Range: {np.min(reaction_orders):.4f} to {np.max(reaction_orders):.4f}")
            print(f"  Average k: {np.mean(rate_constants):.4f} ± {np.std(rate_constants):.4f}")
            print(f"  k Range: {np.min(rate_constants):.4f} to {np.max(rate_constants):.4f}")
            print(f"  Average R²: {np.mean(r_squared_values):.6f}")
            print(f"  R² Range: {np.min(r_squared_values):.6f} to {np.max(r_squared_values):.6f}")
    
    # Create comprehensive summary
    df_results = pd.DataFrame(all_results)
    
    print(f"\n🎯 OVERALL SUMMARY - ALL REACTIONS")
    print(f"{'='*70}")
    
    for rxn in ['R1', 'R2', 'R3']:
        rxn_data = df_results[df_results['Reaction'] == rxn]
        if len(rxn_data) > 0:
            print(f"\n{rxn}:")
            print(f"  Conditions analyzed: {len(rxn_data)}")
            print(f"  Reaction order: {rxn_data['Reaction_Order'].mean():.4f} ± {rxn_data['Reaction_Order'].std():.4f}")
            print(f"  Rate constant range: {rxn_data['Rate_Constant'].min():.4f} - {rxn_data['Rate_Constant'].max():.4f} h⁻¹")
            print(f"  Average R²: {rxn_data['R_Squared'].mean():.6f}")
            
            # Temperature effect on rate constant
            temp_effect = []
            for temp in sorted(rxn_data['Temperature_C'].unique()):
                temp_data = rxn_data[rxn_data['Temperature_C'] == temp]
                avg_k = temp_data['Rate_Constant'].mean()
                temp_effect.append((temp, avg_k))
            
            print(f"  Temperature effect on k:")
            for temp, k in temp_effect:
                print(f"    {temp}°C: k = {k:.4f} h⁻¹")
    
    # Save detailed results
    df_results.to_excel('All_Reactions_Detailed_Analysis.xlsx', index=False)
    print(f"\n💾 Detailed results saved to: All_Reactions_Detailed_Analysis.xlsx")
    
    # Quick comparison table
    print(f"\n📊 QUICK COMPARISON TABLE:")
    print(f"{'Reaction':<8} {'Avg Order':<10} {'Avg k (h⁻¹)':<12} {'k Range':<20} {'Avg R²':<8}")
    print("-" * 70)
    
    for rxn in ['R1', 'R2', 'R3']:
        rxn_data = df_results[df_results['Reaction'] == rxn]
        if len(rxn_data) > 0:
            avg_order = rxn_data['Reaction_Order'].mean()
            avg_k = rxn_data['Rate_Constant'].mean()
            k_min = rxn_data['Rate_Constant'].min()
            k_max = rxn_data['Rate_Constant'].max()
            avg_r2 = rxn_data['R_Squared'].mean()
            
            print(f"{rxn:<8} {avg_order:<10.4f} {avg_k:<12.4f} {k_min:.4f}-{k_max:.4f}    {avg_r2:<8.6f}")
    
    return df_results

if __name__ == "__main__":
    results = analyze_all_reactions()
