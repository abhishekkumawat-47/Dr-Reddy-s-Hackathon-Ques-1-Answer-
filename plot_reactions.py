import pandas as pd
import matplotlib.pyplot as plt
import os

# Set matplotlib parameters for better readability
plt.rcParams.update({
    'font.size': 16,           # Increased base font size
    'axes.titlesize': 20,      # Increased subplot titles
    'axes.labelsize': 18,      # Increased axis labels
    'xtick.labelsize': 16,     # Increased X-axis tick labels
    'ytick.labelsize': 16,     # Increased Y-axis tick labels
    'legend.fontsize': 14,     # Legend
    'figure.titlesize': 22,    # Increased main title
    'lines.linewidth': 3.0,    # Thicker line thickness
    'grid.linewidth': 1.2,     # Grid line thickness
    'axes.linewidth': 1.5,     # Axes border thickness
    'xtick.major.size': 8,     # Larger x-axis tick marks
    'ytick.major.size': 8,     # Larger y-axis tick marks
    'xtick.major.width': 1.5,  # Thicker x-axis tick marks
    'ytick.major.width': 1.5   # Thicker y-axis tick marks
})

# COMPARISON PLOTS

# File paths (adjust if needed)
files = {
    "R1": "Reaction-1.xlsx",
    "R2": "Reaction-2.xlsx",
    "R3": "Reaction-3.xlsx"
}

# Load Calculated sheets
data = {}
for rxn, f in files.items():
    data[rxn] = pd.read_excel(f, sheet_name="Calculated")

# Create output folder
outdir = "comparison_plots"
os.makedirs(outdir, exist_ok=True)

# Define colors
colors = {"R1":"blue", "R2":"red", "R3":"green"}

# Unique (Temp, Conc) pairs (common across reactions)
pairs = sorted(set(zip(data["R1"]["Temp_C"], data["R1"]["A0_mgml"])))

for (T, C0) in pairs:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # Increased size for better spacing
    fig.suptitle(f"Comparison at Temp={T}°C, Conc={C0} mg/mL", fontsize=24, fontweight="bold", y=0.98)

    # Loop reactions
    for rxn in ["R1","R2","R3"]:
        df = data[rxn]
        subset = df[(df["Temp_C"]==T) & (df["A0_mgml"]==C0)]
        
        if subset.empty:
            continue
        
        # Conversion vs Time
        axes[0,0].plot(subset["Time_h"], subset["Conversion_pct"], 
                       label=rxn, color=colors[rxn], linewidth=3.5, marker='o', markersize=8)
        
        # Yield vs Time
        axes[0,1].plot(subset["Time_h"], subset["Yield_pct"], 
                       label=rxn, color=colors[rxn], linewidth=3.5, marker='s', markersize=8)
        
        # Selectivity vs Time
        axes[1,0].plot(subset["Time_h"], subset["Selectivity_pct"], 
                       label=rxn, color=colors[rxn], linewidth=3.5, marker='^', markersize=8)
        
        # Concentrations vs Time (A, B, I)
        axes[1,1].plot(subset["Time_h"], subset["A_mgml"], 
                       label=f"A ({rxn})", linestyle="--", color=colors[rxn], linewidth=3.0)
        axes[1,1].plot(subset["Time_h"], subset["B_mgml"], 
                       label=f"B ({rxn})", linestyle="-", color=colors[rxn], linewidth=3.0)
        axes[1,1].plot(subset["Time_h"], subset["I_mgml"], 
                       label=f"I ({rxn})", linestyle=":", color=colors[rxn], linewidth=3.0)
    
    # Titles & labels with larger fonts
    axes[0,0].set_title("Conversion vs Time", fontsize=20, fontweight='bold', pad=20)
    axes[0,0].set_xlabel("Time (h)", fontsize=18, fontweight='bold')
    axes[0,0].set_ylabel("Conversion (%)", fontsize=18, fontweight='bold')
    
    axes[0,1].set_title("Yield vs Time", fontsize=20, fontweight='bold', pad=20)
    axes[0,1].set_xlabel("Time (h)", fontsize=18, fontweight='bold')
    axes[0,1].set_ylabel("Yield (%)", fontsize=18, fontweight='bold')
    
    axes[1,0].set_title("Selectivity vs Time", fontsize=20, fontweight='bold', pad=20)
    axes[1,0].set_xlabel("Time (h)", fontsize=18, fontweight='bold')
    axes[1,0].set_ylabel("Selectivity (%)", fontsize=18, fontweight='bold')
    
    axes[1,1].set_title("Concentrations vs Time", fontsize=20, fontweight='bold', pad=20)
    axes[1,1].set_xlabel("Time (h)", fontsize=18, fontweight='bold')
    axes[1,1].set_ylabel("Conc (mg/mL)", fontsize=18, fontweight='bold')
    
    # Enhanced legends - BOTTOM RIGHT for first 3 plots, enhanced legend for concentration plot
    # Conversion plot - legend bottom right
    axes[0,0].legend(fontsize=16, frameon=True, fancybox=True, shadow=True, 
                     loc='lower right', facecolor='white', edgecolor='black', framealpha=0.9)
    
    # Yield plot - legend bottom right  
    axes[0,1].legend(fontsize=16, frameon=True, fancybox=True, shadow=True, 
                     loc='lower right', facecolor='white', edgecolor='black', framealpha=0.9)
    
    # Selectivity plot - legend bottom right
    axes[1,0].legend(fontsize=16, frameon=True, fancybox=True, shadow=True, 
                     loc='lower right', facecolor='white', edgecolor='black', framealpha=0.9)
    
    # Concentration plot - larger legend with increased vertical height and fonts
    leg = axes[1,1].legend(fontsize=18, frameon=True, fancybox=True, shadow=True, 
                          bbox_to_anchor=(1.02, 1), loc='upper left', 
                          facecolor='white', edgecolor='black', framealpha=0.9,
                          borderpad=1.2, columnspacing=1.5, handlelength=2.5, 
                          handletextpad=1.0, labelspacing=1.2)
    # Increase legend frame thickness
    leg.get_frame().set_linewidth(2.0)
    
    # Enhanced grids and tick formatting
    for ax in axes.flatten():
        ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.2)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=8)
        ax.tick_params(axis='both', which='minor', labelsize=14, width=1.0, length=6)
        # Add some padding to the axes
        ax.margins(x=0.02, y=0.05)
        # Make axes more prominent
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])  # Adjusted for concentration plot legend
    plt.savefig(f"{outdir}/Comparison_T{T}_C{C0}.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

print(f"## All plots saved in folder: {outdir}")
