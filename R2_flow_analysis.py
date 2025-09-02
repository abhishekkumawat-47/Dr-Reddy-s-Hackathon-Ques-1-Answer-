#!/usr/bin/env python3
"""
Reaction 2 Flow Processing Analysis
==================================
Analyze R2 data to determine optimal conditions for flow processing
with focus on minimizing impurity formation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_R2_data():
    """Load and analyze Reaction 2 data comprehensively"""
    print("REACTION 2 FLOW PROCESSING ANALYSIS")
    print("="*50)
    
    # Load the data
    df = pd.read_excel("Reaction-2.xlsx", sheet_name="Calculated")
    
    print(f"📊 Total data points: {len(df)}")
    print(f"📈 Temperature range: {df['Temp_C'].min()}°C to {df['Temp_C'].max()}°C")
    print(f"📈 Concentration range: {df['A0_mgml'].min()} to {df['A0_mgml'].max()} mg/mL")
    
    return df

def analyze_impurity_formation(df):
    """Analyze impurity formation patterns"""
    print("\n🎯 IMPURITY FORMATION ANALYSIS")
    print("="*40)
    
    # Get final values for each condition (last time point)
    final_conditions = df.groupby(['Temp_C', 'A0_mgml']).last().reset_index()
    
    # Calculate final impurity percentage
    final_conditions['Final_I_Percent'] = (final_conditions['I_mgml'] / final_conditions['A0_mgml']) * 100
    
    # Sort by lowest impurity
    best_conditions = final_conditions.nsmallest(5, 'Final_I_Percent')
    
    print("🏆 TOP 5 CONDITIONS WITH LOWEST IMPURITY:")
    for i, row in best_conditions.iterrows():
        print(f"   {i+1}. {row['Temp_C']}°C, {row['A0_mgml']} mg/mL → {row['Final_I_Percent']:.2f}% impurity")
    
    # Analyze temperature effect
    temp_effect = final_conditions.groupby('Temp_C')['Final_I_Percent'].agg(['mean', 'std'])
    print(f"\n🌡️ TEMPERATURE EFFECT ON IMPURITY:")
    for temp in sorted(final_conditions['Temp_C'].unique()):
        avg_imp = temp_effect.loc[temp, 'mean']
        std_imp = temp_effect.loc[temp, 'std']
        print(f"   {temp}°C: {avg_imp:.2f}% ± {std_imp:.2f}% impurity")
    
    # Analyze concentration effect
    conc_effect = final_conditions.groupby('A0_mgml')['Final_I_Percent'].agg(['mean', 'std'])
    print(f"\n🧪 CONCENTRATION EFFECT ON IMPURITY:")
    for conc in sorted(final_conditions['A0_mgml'].unique()):
        avg_imp = conc_effect.loc[conc, 'mean']
        std_imp = conc_effect.loc[conc, 'std']
        print(f"   {conc} mg/mL: {avg_imp:.2f}% ± {std_imp:.2f}% impurity")
    
    return final_conditions, best_conditions

def analyze_concentration_sensitivity(final_conditions):
    """Determine if reaction is concentration insensitive"""
    print("\n📊 CONCENTRATION SENSITIVITY ANALYSIS")
    print("="*45)
    
    # For each temperature, check how much impurity varies with concentration
    temp_sensitivity = {}
    
    for temp in sorted(final_conditions['Temp_C'].unique()):
        temp_data = final_conditions[final_conditions['Temp_C'] == temp]
        
        # Calculate coefficient of variation (std/mean) for impurity at this temperature
        impurity_values = temp_data['Final_I_Percent']
        cv = impurity_values.std() / impurity_values.mean() * 100
        
        temp_sensitivity[temp] = {
            'cv': cv,
            'min_imp': impurity_values.min(),
            'max_imp': impurity_values.max(),
            'range': impurity_values.max() - impurity_values.min()
        }
        
        print(f"   {temp}°C: CV = {cv:.1f}%, Range = {temp_sensitivity[temp]['range']:.2f}%")
    
    # Overall concentration sensitivity
    overall_cv = np.mean([v['cv'] for v in temp_sensitivity.values()])
    
    print(f"\n🎯 CONCENTRATION SENSITIVITY VERDICT:")
    if overall_cv < 10:
        print(f"   ✅ LOW sensitivity (CV = {overall_cv:.1f}%)")
        print(f"   ✅ Reaction is relatively CONCENTRATION INSENSITIVE")
    elif overall_cv < 25:
        print(f"   ⚠️ MODERATE sensitivity (CV = {overall_cv:.1f}%)")
        print(f"   ⚠️ Some concentration dependence observed")
    else:
        print(f"   ❌ HIGH sensitivity (CV = {overall_cv:.1f}%)")
        print(f"   ❌ Reaction is CONCENTRATION SENSITIVE")
    
    return temp_sensitivity

def analyze_product_B_formation(df):
    """Analyze maximum Product B formation for flow optimization"""
    print("\n🟢 PRODUCT B FORMATION ANALYSIS")
    print("="*40)
    
    # Find maximum B concentration for each condition
    max_B_data = df.groupby(['Temp_C', 'A0_mgml'])['B_mgml'].max().reset_index()
    max_B_data.columns = ['Temp_C', 'A0_mgml', 'Max_B_mgml']
    
    # Calculate B yield
    max_B_data['B_Yield_Percent'] = (max_B_data['Max_B_mgml'] / max_B_data['A0_mgml']) * 100
    
    # Find best conditions for B formation
    best_B_conditions = max_B_data.nlargest(5, 'B_Yield_Percent')
    
    print("🏆 TOP 5 CONDITIONS FOR MAXIMUM PRODUCT B:")
    for i, row in best_B_conditions.iterrows():
        print(f"   {i+1}. {row['Temp_C']}°C, {row['A0_mgml']} mg/mL → {row['B_Yield_Percent']:.1f}% B yield")
    
    return max_B_data

def flow_processing_recommendations(final_conditions, best_conditions, temp_sensitivity):
    """Provide specific flow processing recommendations"""
    print("\n🚀 FLOW PROCESSING RECOMMENDATIONS")
    print("="*45)
    
    # Best overall condition
    best_condition = best_conditions.iloc[0]
    
    print(f"✅ OPTIMAL CONDITIONS FOR FLOW:")
    print(f"   Temperature: {best_condition['Temp_C']}°C")
    print(f"   Concentration: {best_condition['A0_mgml']} mg/mL")
    print(f"   Expected impurity: {best_condition['Final_I_Percent']:.2f}%")
    
    # Temperature recommendations
    low_temp_performance = final_conditions[final_conditions['Temp_C'] <= 40]['Final_I_Percent'].mean()
    high_temp_performance = final_conditions[final_conditions['Temp_C'] >= 60]['Final_I_Percent'].mean()
    
    print(f"\n🌡️ TEMPERATURE STRATEGY:")
    if low_temp_performance < high_temp_performance:
        print(f"   ✅ LOWER temperatures preferred (≤40°C)")
        print(f"   ✅ Average impurity: {low_temp_performance:.2f}% vs {high_temp_performance:.2f}% at high temp")
        print(f"   ✅ Easier heat management in flow systems")
    else:
        print(f"   ⚠️ HIGHER temperatures may be needed (≥60°C)")
        print(f"   ⚠️ Requires better heat management in flow")
    
    # Concentration flexibility
    cv_values = [v['cv'] for v in temp_sensitivity.values()]
    avg_cv = np.mean(cv_values)
    
    print(f"\n🧪 CONCENTRATION FLEXIBILITY:")
    if avg_cv < 15:
        print(f"   ✅ HIGH flexibility - wide concentration range usable")
        print(f"   ✅ Easy to scale up/down in flow systems")
        print(f"   ✅ Robust against concentration variations")
    else:
        print(f"   ⚠️ LIMITED flexibility - concentration control important")
        print(f"   ⚠️ Need precise concentration control in flow")
    
    # Flow-specific advantages
    print(f"\n⚡ FLOW PROCESSING ADVANTAGES:")
    print(f"   ✅ Better heat transfer → precise temperature control")
    print(f"   ✅ Shorter residence time → less impurity formation")
    print(f"   ✅ Continuous operation → consistent quality")
    print(f"   ✅ Easy scale-up with proven conditions")

def create_summary_plot(final_conditions):
    """Create comprehensive summary plot"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Heatmap of impurity vs temperature and concentration
    pivot_imp = final_conditions.pivot(index='Temp_C', columns='A0_mgml', values='Final_I_Percent')
    sns.heatmap(pivot_imp, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                ax=axes[0,0], cbar_kws={'label': 'Impurity %'})
    axes[0,0].set_title('Impurity Formation Heatmap')
    
    # Temperature effect
    temp_avg = final_conditions.groupby('Temp_C')['Final_I_Percent'].mean()
    axes[0,1].plot(temp_avg.index, temp_avg.values, 'o-', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('Temperature (°C)')
    axes[0,1].set_ylabel('Average Impurity (%)')
    axes[0,1].set_title('Temperature Effect on Impurity')
    axes[0,1].grid(True, alpha=0.3)
    
    # Concentration effect
    conc_avg = final_conditions.groupby('A0_mgml')['Final_I_Percent'].mean()
    axes[1,0].plot(conc_avg.index, conc_avg.values, 's-', linewidth=2, markersize=8, color='orange')
    axes[1,0].set_xlabel('Initial Concentration (mg/mL)')
    axes[1,0].set_ylabel('Average Impurity (%)')
    axes[1,0].set_title('Concentration Effect on Impurity')
    axes[1,0].grid(True, alpha=0.3)
    
    # Best conditions scatter
    best_5 = final_conditions.nsmallest(5, 'Final_I_Percent')
    axes[1,1].scatter(final_conditions['Temp_C'], final_conditions['A0_mgml'], 
                     c=final_conditions['Final_I_Percent'], cmap='RdYlGn_r', s=100, alpha=0.7)
    axes[1,1].scatter(best_5['Temp_C'], best_5['A0_mgml'], 
                     c='blue', s=200, marker='*', label='Top 5 Best')
    axes[1,1].set_xlabel('Temperature (°C)')
    axes[1,1].set_ylabel('Concentration (mg/mL)')
    axes[1,1].set_title('Optimal Conditions Map')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('R2_Flow_Processing_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load data
    df = load_and_analyze_R2_data()
    
    # Analyze impurity formation
    final_conditions, best_conditions = analyze_impurity_formation(df)
    
    # Analyze concentration sensitivity
    temp_sensitivity = analyze_concentration_sensitivity(final_conditions)
    
    # Analyze Product B formation
    max_B_data = analyze_product_B_formation(df)
    
    # Provide flow processing recommendations
    flow_processing_recommendations(final_conditions, best_conditions, temp_sensitivity)
    
    # Create summary plot
    create_summary_plot(final_conditions)
    
    print(f"\n🎯 FINAL SUMMARY FOR FLOW PROCESSING:")
    print("="*45)
    best = best_conditions.iloc[0]
    print(f"✅ Best condition: {best['Temp_C']}°C, {best['A0_mgml']} mg/mL")
    print(f"✅ Minimum impurity: {best['Final_I_Percent']:.2f}%")
    print(f"✅ Flow processing highly recommended!")
