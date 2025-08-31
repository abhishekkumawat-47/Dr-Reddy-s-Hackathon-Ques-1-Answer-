import pandas as pd
import matplotlib.pyplot as plt
import os

# Set matplotlib parameters for better readability
plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 16,      # Subplot titles
    'axes.labelsize': 14,      # Axis labels
    'xtick.labelsize': 12,     # X-axis tick labels
    'ytick.labelsize': 12,     # Y-axis tick labels
    'legend.fontsize': 12,     # Legend
    'figure.titlesize': 18,    # Main title
    'lines.linewidth': 2.5,    # Line thickness
    'grid.linewidth': 1.0,     # Grid line thickness
    'axes.linewidth': 1.2      # Axes border thickness
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Increased from (12, 10)
    fig.suptitle(f"Comparison at Temp={T}°C, Conc={C0} mg/mL", fontsize=20, fontweight="bold", y=0.98)

    # Loop reactions
    for rxn in ["R1","R2","R3"]:
        df = data[rxn]
        subset = df[(df["Temp_C"]==T) & (df["A0_mgml"]==C0)]
        
        if subset.empty:
            continue
        
        # Conversion vs Time
        axes[0,0].plot(subset["Time_h"], subset["Conversion_pct"], 
                       label=rxn, color=colors[rxn], linewidth=3, marker='o', markersize=6)
        
        # Yield vs Time
        axes[0,1].plot(subset["Time_h"], subset["Yield_pct"], 
                       label=rxn, color=colors[rxn], linewidth=3, marker='s', markersize=6)
        
        # Selectivity vs Time
        axes[1,0].plot(subset["Time_h"], subset["Selectivity_pct"], 
                       label=rxn, color=colors[rxn], linewidth=3, marker='^', markersize=6)
        
        # Concentrations vs Time (A, B, I)
        axes[1,1].plot(subset["Time_h"], subset["A_mgml"], 
                       label=f"A ({rxn})", linestyle="--", color=colors[rxn], linewidth=2.5)
        axes[1,1].plot(subset["Time_h"], subset["B_mgml"], 
                       label=f"B ({rxn})", linestyle="-", color=colors[rxn], linewidth=2.5)
        axes[1,1].plot(subset["Time_h"], subset["I_mgml"], 
                       label=f"I ({rxn})", linestyle=":", color=colors[rxn], linewidth=2.5)
    
    # Titles & labels with larger fonts
    axes[0,0].set_title("Conversion vs Time", fontsize=16, fontweight='bold', pad=15)
    axes[0,0].set_xlabel("Time (h)", fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel("Conversion (%)", fontsize=14, fontweight='bold')
    
    axes[0,1].set_title("Yield vs Time", fontsize=16, fontweight='bold', pad=15)
    axes[0,1].set_xlabel("Time (h)", fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel("Yield (%)", fontsize=14, fontweight='bold')
    
    axes[1,0].set_title("Selectivity vs Time", fontsize=16, fontweight='bold', pad=15)
    axes[1,0].set_xlabel("Time (h)", fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel("Selectivity (%)", fontsize=14, fontweight='bold')
    
    axes[1,1].set_title("Concentrations vs Time", fontsize=16, fontweight='bold', pad=15)
    axes[1,1].set_xlabel("Time (h)", fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel("Conc (mg/mL)", fontsize=14, fontweight='bold')
    
    # Enhanced legends and grids
    for ax in axes.flatten():
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, 
                 bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, linestyle="--", alpha=0.7, linewidth=1)
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=6)
        # Add some padding to the axes
        ax.margins(x=0.02, y=0.05)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Adjusted for larger legends
    plt.savefig(f"{outdir}/Comparison_T{T}_C{C0}.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

print(f"## All plots saved in folder: {outdir}")
