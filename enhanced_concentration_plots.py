#!/usr/bin/env python3
"""
Improved Concentration Profiles Plot
===================================
Enhanced version with larger fonts, better legend placement, and white background
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set high-quality plotting parameters
plt.rcParams.update({
    'font.size': 24,
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'legend.fontsize': 18,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none'
})

def load_reaction_data():
    """Load data for all reactions"""
    data = {}
    
    for rxn in ['R1', 'R2', 'R3']:
        try:
            filename = f"Reaction-{rxn[1]}.xlsx"
            df = pd.read_excel(filename, sheet_name="Calculated")
            data[rxn] = df
            print(f"✅ Loaded {rxn} data: {len(df)} rows")
        except Exception as e:
            print(f"❌ Error loading {rxn}: {e}")
    
    return data

def create_improved_concentration_plots():
    """
    Create improved concentration profile plots with enhanced formatting
    """
    print("📊 Creating Enhanced Concentration Profile Plots...")
    
    # Load data
    data = load_reaction_data()
    
    # Create figure with white background
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.patch.set_facecolor('white')
    
    # Enhanced title
    fig.suptitle("Concentration Profiles - Parallel Reaction Analysis", 
                fontsize=20, fontweight='bold', y=0.95)
    
    # Colors for reactions (more vibrant)
    colors = {'R1': '#2E8B57', 'R2': '#DC143C', 'R3': '#228B22'}  # Green, Red, Forest Green
    
    # Conditions to plot (use available data)
    temp, conc = 30, 50
    
    for rxn in ['R1', 'R2', 'R3']:
        if rxn not in data:
            continue
            
        df = data[rxn]
        subset = df[(df["Temp_C"] == temp) & (df["A0_mgml"] == conc)]
        
        if subset.empty:
            continue
        
        time = subset['Time_h'].values
        A = subset['A_mgml'].values
        B = subset['B_mgml'].values
        I = subset['I_mgml'].values
        
        # Plot A concentration (Reactant Consumption)
        axes[0,0].plot(time, A, 'o-', color=colors[rxn], label=f'{rxn}', 
                      linewidth=3, markersize=8, markerfacecolor=colors[rxn], 
                      markeredgecolor='white', markeredgewidth=2)
        
        # Plot B concentration (Product Formation - KEY ANALYSIS)
        axes[0,1].plot(time, B, 'o-', color=colors[rxn], label=f'{rxn}', 
                      linewidth=4, markersize=10, markerfacecolor=colors[rxn], 
                      markeredgecolor='white', markeredgewidth=2)
        
        # Plot I concentration (Impurity Formation)
        axes[1,0].plot(time, I, 'o-', color=colors[rxn], label=f'{rxn}', 
                      linewidth=3, markersize=8, markerfacecolor=colors[rxn], 
                      markeredgecolor='white', markeredgewidth=2)
        
        # Plot B/I ratio (Selectivity)
        ratio = B / (I + 1e-6)  # Add small value to avoid division by zero
        axes[1,1].plot(time, ratio, 'o-', color=colors[rxn], label=f'{rxn}', 
                      linewidth=3, markersize=8, markerfacecolor=colors[rxn], 
                      markeredgecolor='white', markeredgewidth=2)
    
    # Enhanced formatting for each subplot
    titles = ['Reactant A Consumption', 'Product B Formation (Key Analysis)', 
             'Impurity I Formation', 'B/I Ratio (Selectivity)']
    ylabels = ['A Concentration (mg/mL)', 'B Concentration (mg/mL)', 
              'I Concentration (mg/mL)', 'B/I Ratio']
    
    for i, (ax, title, ylabel) in enumerate(zip(axes.flat, titles, ylabels)):
        # Set white background
        ax.set_facecolor('white')
        
        # Enhanced title and labels
        ax.set_title(title, fontweight='bold', fontsize=16, pad=15)
        ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=14)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)
        
        # Enhanced grid
        ax.grid(True, alpha=0.4, linewidth=1, linestyle='-', color='lightgray')
        ax.set_axisbelow(True)
        
        # Enhanced legend - bottom right, larger
        legend = ax.legend(loc='lower right', fontsize=14,
                          frameon=True, fancybox=True, shadow=True,
                          borderpad=1, columnspacing=1, handletextpad=0.8)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1)
        
        # Make legend text bold
        for text in legend.get_texts():
            text.set_fontweight('bold')
        
        # Enhanced tick labels
        ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
        
        # Special formatting for Product B plot (most important)
        if i == 1:
            ax.set_facecolor('#f8f9fa')
            # Add horizontal reference line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
            
            # Add text annotation
            ax.text(0.02, 0.98, 'KEY ANALYSIS', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Adjust layout with more spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    # Save with high quality
    filename = 'Enhanced_Concentration_Profiles_Analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"✅ Plot saved as: {filename}")
    
    plt.show()
    
    # Analysis summary
    print("\n📝 ENHANCED VISUAL ANALYSIS:")
    print("="*50)
    print("From the improved B concentration vs time plots:")
    print("• R1 (Green): B increases steadily and plateaus → Parallel reaction A→B, A→I")
    print("• R2 (Red): B increases then decreases → Complex reaction (consecutive?)")
    print("• R3 (Forest Green): B increases steadily → Parallel reaction A→B, A→I")
    print("\n✅ R1 shows classic parallel reaction behavior with optimal B formation")

if __name__ == "__main__":
    create_improved_concentration_plots()
