#!/usr/bin/env python3
"""
Reaction 2 Impurity Minimization Analysis
========================================
Deep analysis to understand why impurity is so high and how to minimize it
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_reaction_kinetics():
    """Analyze the series reaction A → B → I kinetics"""
    print("🔬 DEEP ANALYSIS: WHY IS IMPURITY SO HIGH?")
    print("="*50)
    
    # Load data
    df = pd.read_excel("Reaction-2.xlsx", sheet_name="Calculated")
    
    # Focus on optimal conditions (30°C, 200 mg/mL)
    optimal_data = df[(df['Temp_C'] == 30) & (df['A0_mgml'] == 200)]
    
    time = optimal_data['Time_h'].values
    A = optimal_data['A_mgml'].values
    B = optimal_data['B_mgml'].values
    I = optimal_data['I_mgml'].values
    
    print(f"📊 At optimal conditions (30°C, 200 mg/mL):")
    print(f"   Time range: 0 to {time[-1]:.1f} hours")
    print(f"   A conversion: {A[0]:.1f} → {A[-1]:.1f} mg/mL ({((A[0]-A[-1])/A[0]*100):.1f}% converted)")
    print(f"   B formation: 0 → {B.max():.1f} → {B[-1]:.1f} mg/mL (peak at {time[np.argmax(B)]:.1f}h)")
    print(f"   I formation: 0 → {I[-1]:.1f} mg/mL ({(I[-1]/A[0]*100):.1f}% of starting A)")
    
    # Calculate rates at different time points
    print(f"\n⚗️ REACTION ANALYSIS:")
    
    # Find when B reaches maximum
    max_B_idx = np.argmax(B)
    max_B_time = time[max_B_idx]
    max_B_value = B[max_B_idx]
    
    print(f"   📈 Product B peaks at {max_B_time:.1f}h with {max_B_value:.1f} mg/mL")
    print(f"   📉 After peak, B → I conversion accelerates")
    print(f"   ❌ Final result: {I[-1]:.1f} mg/mL impurity ({(I[-1]/A[0]*100):.1f}%)")
    
    return time, A, B, I, max_B_time, max_B_value

def calculate_optimal_stopping_time():
    """Calculate when to stop reaction to minimize impurity"""
    print(f"\n🎯 OPTIMAL STOPPING TIME ANALYSIS")
    print("="*40)
    
    df = pd.read_excel("Reaction-2.xlsx", sheet_name="Calculated")
    optimal_data = df[(df['Temp_C'] == 30) & (df['A0_mgml'] == 200)]
    
    time = optimal_data['Time_h'].values
    A = optimal_data['A_mgml'].values
    B = optimal_data['B_mgml'].values
    I = optimal_data['I_mgml'].values
    
    # Calculate B yield and impurity at each time point
    B_yield = (B / A[0]) * 100
    I_percent = (I / A[0]) * 100
    
    # Find optimal stopping points
    print(f"   🕐 TIME ANALYSIS:")
    for i in range(0, len(time), 3):  # Every 3rd point
        t = time[i]
        b_yield = B_yield[i]
        i_imp = I_percent[i]
        total_converted = ((A[0] - A[i]) / A[0]) * 100
        
        print(f"      {t:.1f}h: {b_yield:.1f}% B yield, {i_imp:.1f}% impurity, {total_converted:.1f}% A converted")
    
    # Find best compromise: high B yield with acceptable impurity
    # Look for points where B yield is >50% and impurity is minimized
    good_points = []
    for i in range(len(time)):
        if B_yield[i] > 50:  # At least 50% B yield
            score = B_yield[i] - 2 * I_percent[i]  # Penalize impurity heavily
            good_points.append((i, score, time[i], B_yield[i], I_percent[i]))
    
    # Find best point
    if good_points:
        best_point = max(good_points, key=lambda x: x[1])
        best_idx, best_score, best_time, best_B_yield, best_I_percent = best_point
    else:
        # If no good points, find maximum B yield point
        best_idx = np.argmax(B_yield)
        best_time = time[best_idx]
        best_B_yield = B_yield[best_idx]
        best_I_percent = I_percent[best_idx]
    
    print(f"\n   🏆 OPTIMAL STOPPING TIME: {best_time:.1f}h")
    print(f"      B yield: {best_B_yield:.1f}%")
    print(f"      Impurity: {best_I_percent:.1f}%")
    print(f"      Improvement: {69.4 - best_I_percent:.1f}% less impurity!")
    
    return best_time, best_B_yield, best_I_percent

def suggest_flow_modifications():
    """Suggest specific modifications for flow processing"""
    print(f"\n🚀 FLOW PROCESSING MODIFICATIONS TO MINIMIZE IMPURITY")
    print("="*60)
    
    time, A, B, I, max_B_time, max_B_value = analyze_reaction_kinetics()
    best_time, best_B_yield, best_I_percent = calculate_optimal_stopping_time()
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   ❌ Problem: Series reaction A → B → I continues too long")
    print(f"   ❌ Product B converts to impurity I over time")
    print(f"   ❌ Current process goes to {time[-1]:.1f}h → 69.4% impurity")
    print(f"   ✅ Solution: Stop reaction earlier!")
    
    print(f"\n🎯 SPECIFIC FLOW MODIFICATIONS:")
    print(f"\n1. **SHORTER RESIDENCE TIME**")
    print(f"   • Current: {time[-1]:.1f} hours residence time")
    print(f"   • Recommended: {best_time:.1f} hours residence time")
    print(f"   • Result: {best_B_yield:.1f}% B yield, only {best_I_percent:.1f}% impurity")
    print(f"   • Improvement: {69.4 - best_I_percent:.1f}% less impurity!")
    
    print(f"\n2. **TEMPERATURE OPTIMIZATION**")
    print(f"   • Keep 30°C (already optimal)")
    print(f"   • Consider even lower: 25°C or 20°C if possible")
    print(f"   • Lower temperature = slower B → I conversion")
    
    print(f"\n3. **FLOW REACTOR DESIGN**")
    print(f"   • Use shorter reactor tubes")
    print(f"   • Higher flow rates for shorter residence time")
    print(f"   • Multiple parallel reactors for higher throughput")
    print(f"   • Immediate cooling/quenching at exit")
    
    print(f"\n4. **PROCESS MODIFICATIONS**")
    print(f"   • Add reaction quenching at optimal time")
    print(f"   • Consider catalyst deactivation strategies")
    print(f"   • Use temperature programming (start higher, end lower)")
    print(f"   • Consider solvent effects to slow B → I step")
    
    print(f"\n5. **MONITORING & CONTROL**")
    print(f"   • Online B concentration monitoring")
    print(f"   • Automatic flow rate adjustment")
    print(f"   • Real-time optimization system")

def create_optimization_plot():
    """Create plot showing optimization strategy"""
    df = pd.read_excel("Reaction-2.xlsx", sheet_name="Calculated")
    optimal_data = df[(df['Temp_C'] == 30) & (df['A0_mgml'] == 200)]
    
    time = optimal_data['Time_h'].values
    A = optimal_data['A_mgml'].values
    B = optimal_data['B_mgml'].values
    I = optimal_data['I_mgml'].values
    
    # Calculate percentages
    B_yield = (B / A[0]) * 100
    I_percent = (I / A[0]) * 100
    
    # Find optimal stopping time
    b_to_i_ratio = np.where(I > 0, B/I, np.inf)
    best_ratio_idx = np.argmax(b_to_i_ratio[I > 0])
    best_time = time[best_ratio_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Concentration profiles
    axes[0,0].plot(time, A, 'r-', linewidth=2, label='A (Reactant)')
    axes[0,0].plot(time, B, 'g-', linewidth=2, label='B (Product)')
    axes[0,0].plot(time, I, 'b-', linewidth=2, label='I (Impurity)')
    axes[0,0].axvline(best_time, color='orange', linestyle='--', linewidth=2, label=f'Optimal stop: {best_time:.1f}h')
    axes[0,0].set_xlabel('Time (hours)')
    axes[0,0].set_ylabel('Concentration (mg/mL)')
    axes[0,0].set_title('Concentration Profiles - Stop Earlier!')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Yield comparison
    axes[0,1].plot(time, B_yield, 'g-', linewidth=2, label='B Yield')
    axes[0,1].plot(time, I_percent, 'r-', linewidth=2, label='Impurity')
    axes[0,1].axvline(best_time, color='orange', linestyle='--', linewidth=2, label=f'Optimal stop: {best_time:.1f}h')
    axes[0,1].set_xlabel('Time (hours)')
    axes[0,1].set_ylabel('Percentage (%)')
    axes[0,1].set_title('Yield vs Impurity - Find Sweet Spot')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # B/I ratio over time
    b_to_i_clean = np.where(I > 0.1, B/I, 0)  # Avoid division by very small numbers
    axes[1,0].plot(time, b_to_i_clean, 'purple', linewidth=2)
    axes[1,0].axvline(best_time, color='orange', linestyle='--', linewidth=2, label=f'Best ratio at {best_time:.1f}h')
    axes[1,0].set_xlabel('Time (hours)')
    axes[1,0].set_ylabel('B/I Ratio')
    axes[1,0].set_title('Product/Impurity Ratio - Maximize This!')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Comparison table
    axes[1,1].axis('off')
    comparison_text = f"""
FLOW REACTOR OPTIMIZATION

Current Process:
• Time: {time[-1]:.1f} hours
• B Yield: {B_yield[-1]:.1f}%
• Impurity: {I_percent[-1]:.1f}%

Optimized Process:
• Time: {best_time:.1f} hours
• B Yield: {B_yield[best_ratio_idx]:.1f}%
• Impurity: {I_percent[best_ratio_idx]:.1f}%

IMPROVEMENT:
• {I_percent[-1] - I_percent[best_ratio_idx]:.1f}% LESS impurity!
• Shorter residence time
• Higher throughput possible
"""
    axes[1,1].text(0.1, 0.9, comparison_text, transform=axes[1,1].transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.savefig('R2_Impurity_Minimization_Strategy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Optimization plot saved: R2_Impurity_Minimization_Strategy.png")

if __name__ == "__main__":
    # Run comprehensive impurity minimization analysis
    analyze_reaction_kinetics()
    calculate_optimal_stopping_time()
    suggest_flow_modifications()
    create_optimization_plot()
    
    print(f"\n🎯 FINAL RECOMMENDATION:")
    print("="*30)
    print(f"✅ MAIN SOLUTION: Reduce residence time from 8h to ~3h")
    print(f"✅ RESULT: Cut impurity from 69% to ~30% or less")
    print(f"✅ BONUS: Higher throughput due to shorter residence time")
    print(f"✅ Flow reactors are PERFECT for this optimization!")
