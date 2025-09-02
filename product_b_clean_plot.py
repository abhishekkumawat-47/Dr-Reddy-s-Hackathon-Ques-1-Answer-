#!/usr/bin/env python3
"""
Product B Formation Plot Only - Clean Version
=============================================
Single plot for Product B Formation with no grids and pure white background
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set high-quality plotting parameters - no grids
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
    'savefig.edgecolor': 'none',
    'axes.grid': False  # Ensure no grids
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

def create_product_b_plot_only():
    """
    Create clean Product B Formation plot only - no grids, white background
    """
    print("📊 Creating Clean Product B Formation Plot...")
    
    # Load data
    data = load_reaction_data()
    
    # Create single figure with pure white background
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Colors for reactions (vibrant)
    colors = {'R1': '#2E8B57', 'R2': '#DC143C', 'R3': '#228B22'}
    
    # Conditions to plot
    temp, conc = 30, 50
    
    for rxn in ['R1', 'R2', 'R3']:
        if rxn not in data:
            continue
            
        df = data[rxn]
        subset = df[(df["Temp_C"] == temp) & (df["A0_mgml"] == conc)]
        
        if subset.empty:
            continue
        
        time = subset['Time_h'].values
        B = subset['B_mgml'].values
        
        # Plot B concentration (Product Formation - KEY ANALYSIS)
        ax.plot(time, B, 'o-', color=colors[rxn], label=f'{rxn}', 
               linewidth=4, markersize=10, markerfacecolor=colors[rxn], 
               markeredgecolor='white', markeredgewidth=2)
    
    # Clean formatting - NO GRIDS
    ax.set_title('Product B Formation (Key Analysis)', fontweight='bold', fontsize=20, pad=20)
    ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=16)
    ax.set_ylabel('B Concentration (mg/mL)', fontweight='bold', fontsize=16)
    
    # NO GRIDS - completely clean background
    ax.grid(False)
    ax.set_axisbelow(False)
    
    # Enhanced legend - bottom right, larger
    legend = ax.legend(loc='lower right', fontsize=18,
                      frameon=True, fancybox=True, shadow=True,
                      borderpad=1.5, columnspacing=1, handletextpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)  # Fully opaque white
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(2)
    
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    # Enhanced tick labels
    ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    
    # Clean white background - no special effects
    ax.set_facecolor('white')
    
    # Remove all spines or make them very subtle
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high quality - pure white background
    filename = 'Product_B_Formation_Clean.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"✅ Clean plot saved as: {filename}")
    
    plt.show()
    
    print("\n📝 CLEAN VISUAL ANALYSIS:")
    print("="*40)
    print("Product B Formation Analysis:")
    print("• R1 (Green): Optimal B formation with plateau")
    print("• R2 (Red): Peak then decline - not ideal")
    print("• R3 (Forest Green): Steady formation")
    print("\n✅ Clean, grid-free visualization complete!")

if __name__ == "__main__":
    create_product_b_plot_only()
