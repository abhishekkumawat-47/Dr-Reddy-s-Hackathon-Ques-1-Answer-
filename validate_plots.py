import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=== PLOT VALIDATION REPORT ===\n")

# Load data
files = {
    "R1": "Reaction-1.xlsx",
    "R2": "Reaction-2.xlsx", 
    "R3": "Reaction-3.xlsx"
}

data = {}
for rxn, f in files.items():
    data[rxn] = pd.read_excel(f, sheet_name="Calculated")
    print(f"{rxn}: Loaded {len(data[rxn])} data points from {f}")

print("\n=== DATA CONSISTENCY CHECKS ===")

# 1. Check if all reactions have the same conditions
for rxn in ["R1", "R2", "R3"]:
    pairs = sorted(set(zip(data[rxn]["Temp_C"], data[rxn]["A0_mgml"])))
    print(f"{rxn}: {len(pairs)} unique (Temp, Conc) combinations")

# 2. Check for missing data
print("\n=== MISSING DATA CHECK ===")
for rxn in ["R1", "R2", "R3"]:
    df = data[rxn]
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"{rxn}: Missing values found!")
        print(missing[missing > 0])
    else:
        print(f"{rxn}: No missing values")

# 3. Check data ranges for reasonableness
print("\n=== DATA RANGE VALIDATION ===")
for rxn in ["R1", "R2", "R3"]:
    df = data[rxn]
    print(f"\n{rxn} Data Ranges:")
    print(f"  Conversion: {df['Conversion_pct'].min():.1f}% - {df['Conversion_pct'].max():.1f}%")
    print(f"  Yield: {df['Yield_pct'].min():.1f}% - {df['Yield_pct'].max():.1f}%")
    print(f"  Selectivity: {df['Selectivity_pct'].min():.1f}% - {df['Selectivity_pct'].max():.1f}%")
    print(f"  Time: {df['Time_h'].min():.1f}h - {df['Time_h'].max():.1f}h")

# 4. Check mass balance (A + B + I should approximately equal A0)
print("\n=== MASS BALANCE CHECK ===")
for rxn in ["R1", "R2", "R3"]:
    df = data[rxn]
    total_mass = df['A_mgml'] + df['B_mgml'] + df['I_mgml']
    mass_balance_error = abs(total_mass - df['A0_mgml'])
    max_error = mass_balance_error.max()
    avg_error = mass_balance_error.mean()
    print(f"{rxn}: Max mass balance error: {max_error:.3f} mg/mL, Average: {avg_error:.3f} mg/mL")

# 5. Sample specific condition for detailed check
print("\n=== DETAILED CHECK FOR T=50°C, C=100mg/mL ===")
T, C = 50, 100

for rxn in ["R1", "R2", "R3"]:
    df = data[rxn]
    subset = df[(df["Temp_C"]==T) & (df["A0_mgml"]==C)]
    if not subset.empty:
        print(f"\n{rxn} at {T}°C, {C}mg/mL:")
        print(f"  Time points: {len(subset)} ({subset['Time_h'].min():.1f}h to {subset['Time_h'].max():.1f}h)")
        print(f"  Final conversion: {subset['Conversion_pct'].iloc[-1]:.1f}%")
        print(f"  Final yield: {subset['Yield_pct'].iloc[-1]:.1f}%")
        print(f"  Final selectivity: {subset['Selectivity_pct'].iloc[-1]:.1f}%")
        print(f"  A concentration: {subset['A_mgml'].iloc[0]:.1f} → {subset['A_mgml'].iloc[-1]:.1f} mg/mL")
        print(f"  B concentration: {subset['B_mgml'].iloc[0]:.1f} → {subset['B_mgml'].iloc[-1]:.1f} mg/mL")

# 6. Check for logical consistency
print("\n=== LOGICAL CONSISTENCY CHECKS ===")
for rxn in ["R1", "R2", "R3"]:
    df = data[rxn]
    
    # Conversion should generally increase with time
    issues = []
    for (T, C) in sorted(set(zip(df["Temp_C"], df["A0_mgml"]))):
        subset = df[(df["Temp_C"]==T) & (df["A0_mgml"]==C)].sort_values('Time_h')
        if len(subset) > 1:
            # Check if conversion is mostly increasing
            conv_diffs = subset['Conversion_pct'].diff().dropna()
            decreasing_points = (conv_diffs < -1).sum()  # Allow small decreases (measurement error)
            if decreasing_points > len(conv_diffs) * 0.2:  # More than 20% decreasing
                issues.append(f"T={T}°C, C={C}mg/mL: Conversion decreases in {decreasing_points}/{len(conv_diffs)} intervals")
    
    if issues:
        print(f"{rxn}: Potential issues found:")
        for issue in issues[:3]:  # Show first 3 issues
            print(f"  - {issue}")
    else:
        print(f"{rxn}: No major logical inconsistencies found")

print("\n=== VALIDATION COMPLETE ===")
print("Check the generated plots in 'comparison_plots' folder and compare with this data summary.")
print("Look for:")
print("1. Curves should start at expected initial values")
print("2. Conversion should generally increase with time")
print("3. Higher temperatures should generally show faster reactions")
print("4. Mass balance should be maintained (A + B + I ≈ A0)")
print("5. Selectivity = Yield/Conversion should make sense")
