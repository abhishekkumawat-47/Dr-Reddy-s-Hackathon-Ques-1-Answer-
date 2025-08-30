import pandas as pd
import matplotlib.pyplot as plt
import os


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
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Comparison at Temp={T}°C, Conc={C0} mg/mL", fontsize=14, fontweight="bold")

    # Loop reactions
    for rxn in ["R1","R2","R3"]:
        df = data[rxn]
        subset = df[(df["Temp_C"]==T) & (df["A0_mgml"]==C0)]
        
        if subset.empty:
            continue
        
        # Conversion vs Time
        axes[0,0].plot(subset["Time_h"], subset["Conversion_pct"], 
                       label=rxn, color=colors[rxn])
        
        # Yield vs Time
        axes[0,1].plot(subset["Time_h"], subset["Yield_pct"], 
                       label=rxn, color=colors[rxn])
        
        # Selectivity vs Time
        axes[1,0].plot(subset["Time_h"], subset["Selectivity_pct"], 
                       label=rxn, color=colors[rxn])
        
        # Concentrations vs Time (A, B, I)
        axes[1,1].plot(subset["Time_h"], subset["A_mgml"], 
                       label=f"A ({rxn})", linestyle="--", color=colors[rxn])
        axes[1,1].plot(subset["Time_h"], subset["B_mgml"], 
                       label=f"B ({rxn})", linestyle="-", color=colors[rxn])
        axes[1,1].plot(subset["Time_h"], subset["I_mgml"], 
                       label=f"I ({rxn})", linestyle=":", color=colors[rxn])
    
    # Titles & labels
    axes[0,0].set_title("Conversion vs Time"); axes[0,0].set_xlabel("Time (h)"); axes[0,0].set_ylabel("Conversion (%)")
    axes[0,1].set_title("Yield vs Time"); axes[0,1].set_xlabel("Time (h)"); axes[0,1].set_ylabel("Yield (%)")
    axes[1,0].set_title("Selectivity vs Time"); axes[1,0].set_xlabel("Time (h)"); axes[1,0].set_ylabel("Selectivity (%)")
    axes[1,1].set_title("Concentrations vs Time"); axes[1,1].set_xlabel("Time (h)"); axes[1,1].set_ylabel("Conc (mg/mL)")
    
    # Legends
    for ax in axes.flatten():
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{outdir}/Comparison_T{T}_C{C0}.png", dpi=300)
    plt.close()

print(f"## All plots saved in folder: {outdir}")
