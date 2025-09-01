import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def determine_exact_reaction_order():
    """
    Determine exact reaction order using variable order kinetics:
    [A] = [A₀] * (1 + (n-1)*k*[A₀]^(n-1)*t)^(-1/(n-1))  for n ≠ 1
    [A] = [A₀] * exp(-kt) for n = 1
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
    
    results = {}
    
    for rxn, file in files.items():
        print(f"\n🔬 Analyzing {rxn}...")
        
        # Load data
        df = pd.read_excel(file, sheet_name="Calculated")
        
        # Get one representative condition for detailed analysis
        temp, conc = 50, 100  # mg/mL
        subset = df[(df["Temp_C"] == temp) & (df["A0_mgml"] == conc)]
        
        if len(subset) < 5:
            print(f"❌ Insufficient data for {rxn}")
            continue
            
        time_data = subset['Time_h'].values
        conc_data = subset['A_mgml'].values
        
        # Remove invalid data
        valid_mask = (conc_data > 0) & (time_data >= 0)
        time_data = time_data[valid_mask]
        conc_data = conc_data[valid_mask]
        
        if len(time_data) < 5:
            print(f"❌ Insufficient valid data for {rxn}")
            continue
            
        A0_initial = conc_data[0]
        
        print(f"  📊 Data points: {len(time_data)}")
        print(f"  🧪 Initial [A]: {A0_initial:.2f} mg/mL")
        
        # Try to fit general order model
        try:
            # Initial guess: A0, k, n
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
            
            # Calculate standard error of n
            param_errors = np.sqrt(np.diag(pcov))
            n_error = param_errors[2]
            
            results[rxn] = {
                'reaction_order': n_fit,
                'order_error': n_error,
                'rate_constant': k_fit,
                'r_squared': r_squared,
                'A0_fit': A0_fit
            }
            
            print(f"  ✅ Reaction Order: {n_fit:.4f} ± {n_error:.4f}")
            print(f"  📈 Rate Constant: {k_fit:.4f}")
            print(f"  📊 R²: {r_squared:.6f}")
            
            # Verify with integer orders
            orders_to_test = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            print(f"  🔍 Comparison with standard orders:")
            
            for test_order in orders_to_test:
                try:
                    if abs(test_order - 1.0) < 1e-6:
                        # First order
                        def test_model(t, A0, k):
                            return A0 * np.exp(-k * t)
                        popt_test, _ = curve_fit(test_model, time_data, conc_data, 
                                               p0=[A0_initial, 0.1], maxfev=5000)
                        y_pred_test = test_model(time_data, *popt_test)
                    else:
                        # Fixed order
                        def test_model(t, A0, k):
                            return general_order_model(t, A0, k, test_order)
                        popt_test, _ = curve_fit(test_model, time_data, conc_data, 
                                               p0=[A0_initial, 0.1], maxfev=5000)
                        y_pred_test = test_model(time_data, *popt_test)
                    
                    ss_res_test = np.sum((conc_data - y_pred_test) ** 2)
                    r2_test = 1 - (ss_res_test / ss_tot)
                    
                    print(f"    Order {test_order}: R² = {r2_test:.6f}")
                    
                except:
                    print(f"    Order {test_order}: Failed to fit")
            
        except Exception as e:
            print(f"❌ Failed to fit {rxn}: {e}")
            
            # Fallback to first-order analysis
            try:
                def first_order(t, A0, k):
                    return A0 * np.exp(-k * t)
                
                popt_fo, _ = curve_fit(first_order, time_data, conc_data, 
                                     p0=[A0_initial, 0.1], maxfev=5000)
                y_pred_fo = first_order(time_data, *popt_fo)
                ss_res_fo = np.sum((conc_data - y_pred_fo) ** 2)
                ss_tot = np.sum((conc_data - np.mean(conc_data)) ** 2)
                r2_fo = 1 - (ss_res_fo / ss_tot)
                
                results[rxn] = {
                    'reaction_order': 1.0,
                    'order_error': 0.0,
                    'rate_constant': popt_fo[1],
                    'r_squared': r2_fo,
                    'A0_fit': popt_fo[0]
                }
                
                print(f"  ✅ Fallback - First Order: R² = {r2_fo:.6f}")
                
            except:
                print(f"❌ Complete failure for {rxn}")
    
    print(f"\n🎯 FINAL REACTION ORDERS:")
    print("="*60)
    for rxn in ['R1', 'R2', 'R3']:
        if rxn in results:
            order = results[rxn]['reaction_order']
            error = results[rxn]['order_error']
            r2 = results[rxn]['r_squared']
            print(f"{rxn}: {order:.4f} ± {error:.4f} (R² = {r2:.6f})")
        else:
            print(f"{rxn}: Analysis failed")
    
    return results

if __name__ == "__main__":
    results = determine_exact_reaction_order()
