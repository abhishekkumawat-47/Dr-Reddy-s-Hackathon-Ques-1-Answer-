import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def load_reaction_data():
    """Load all reaction data"""
    files = {
        "R1": "Reaction-1.xlsx",
        "R2": "Reaction-2.xlsx", 
        "R3": "Reaction-3.xlsx"
    }
    
    data = {}
    for rxn, f in files.items():
        try:
            data[rxn] = pd.read_excel(f, sheet_name="Calculated")
            print(f"✓ Loaded {rxn}: {len(data[rxn])} data points from {f}")
        except FileNotFoundError:
            print(f"✗ Error: Could not find {f}")
            return None
    return data

def get_available_conditions(data):
    """Get all available temperature and concentration combinations"""
    conditions = set()
    for rxn_data in data.values():
        for _, row in rxn_data.iterrows():
            conditions.add((row['Temp_C'], row['A0_mgml']))
    return sorted(list(conditions))

def analyze_condition(data, temperature, concentration, save_plot=True, show_plot=True):
    """Analyze a specific temperature and concentration condition"""
    
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: T={temperature}°C, Conc={concentration} mg/mL")
    print(f"{'='*80}\n")
    
    # Extract specific condition data
    reaction_data = {}
    for rxn in ["R1", "R2", "R3"]:
        df = data[rxn]
        subset = df[(df["Temp_C"]==temperature) & (df["A0_mgml"]==concentration)].sort_values('Time_h')
        reaction_data[rxn] = subset
        
        if subset.empty:
            print(f"⚠️  Warning: No data found for {rxn} at T={temperature}°C, C={concentration}mg/mL")
            continue
            
        print(f"=== {rxn} ANALYSIS ===")
        print(f"Number of time points: {len(subset)}")
        print(f"Time range: {subset['Time_h'].min():.1f}h to {subset['Time_h'].max():.1f}h")
        
        # Initial vs Final conditions
        initial = subset.iloc[0]
        final = subset.iloc[-1]
        
        print(f"\nInitial → Final:")
        print(f"  A: {initial['A_mgml']:.1f} → {final['A_mgml']:.1f} mg/mL")
        print(f"  B: {initial['B_mgml']:.1f} → {final['B_mgml']:.1f} mg/mL")
        print(f"  I: {initial['I_mgml']:.1f} → {final['I_mgml']:.1f} mg/mL")
        print(f"  Conversion: {initial['Conversion_pct']:.1f} → {final['Conversion_pct']:.1f}%")
        print(f"  Yield: {initial['Yield_pct']:.1f} → {final['Yield_pct']:.1f}%")
        
        final_sel = final['Selectivity_pct'] if pd.notna(final['Selectivity_pct']) else 0
        print(f"  Selectivity: {final_sel:.1f}%")
        
        # Mass balance
        total_final = final['A_mgml'] + final['B_mgml'] + final['I_mgml']
        mass_error = abs(total_final - initial['A0_mgml'])
        print(f"\nMass Balance: {total_final:.1f} mg/mL (error: {mass_error:.3f} mg/mL)")
        
        # Reaction rate
        if len(subset) > 1:
            avg_rate = (final['Conversion_pct'] - initial['Conversion_pct']) / (final['Time_h'] - initial['Time_h'])
            print(f"Average conversion rate: {avg_rate:.1f}%/h")
        
        print(f"\n{'-'*50}")
    
    # Comparative summary
    print(f"\n=== COMPARATIVE SUMMARY ===")
    summary_data = []
    
    for rxn in ["R1", "R2", "R3"]:
        subset = reaction_data[rxn]
        if not subset.empty:
            final = subset.iloc[-1]
            final_sel = final['Selectivity_pct'] if pd.notna(final['Selectivity_pct']) else 0
            
            summary_data.append({
                'Reaction': rxn,
                'Time_max': subset['Time_h'].max(),
                'Final_Conv': final['Conversion_pct'],
                'Final_Yield': final['Yield_pct'],
                'Final_Sel': final_sel,
                'Final_A': final['A_mgml'],
                'Final_B': final['B_mgml'],
                'Final_I': final['I_mgml']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\nPerformance Comparison:")
        print(f"{'Reaction':<10} {'Time(h)':<8} {'Conv(%)':<8} {'Yield(%)':<9} {'Sel(%)':<8} {'A_final':<8} {'B_final':<8} {'I_final':<8}")
        print("-" * 75)
        
        for _, row in summary_df.iterrows():
            print(f"{row['Reaction']:<10} {row['Time_max']:<8.1f} {row['Final_Conv']:<8.1f} {row['Final_Yield']:<9.1f} {row['Final_Sel']:<8.1f} {row['Final_A']:<8.1f} {row['Final_B']:<8.1f} {row['Final_I']:<8.1f}")
        
        # Best performers
        best_yield_idx = summary_df['Final_Yield'].idxmax()
        best_sel_idx = summary_df['Final_Sel'].idxmax()
        fastest_idx = summary_df['Time_max'].idxmin()
        
        print(f"\nBest Performers:")
        print(f"  🏆 Highest Yield: {summary_df.loc[best_yield_idx, 'Reaction']} ({summary_df.loc[best_yield_idx, 'Final_Yield']:.1f}%)")
        print(f"  🎯 Best Selectivity: {summary_df.loc[best_sel_idx, 'Reaction']} ({summary_df.loc[best_sel_idx, 'Final_Sel']:.1f}%)")
        print(f"  ⚡ Fastest: {summary_df.loc[fastest_idx, 'Reaction']} ({summary_df.loc[fastest_idx, 'Time_max']:.1f}h)")
    
    # Create detailed plot
    if show_plot or save_plot:
        create_detailed_plot(reaction_data, temperature, concentration, save_plot, show_plot)
    
    return reaction_data, summary_data

def create_detailed_plot(reaction_data, temperature, concentration, save_plot=True, show_plot=True):
    """Create detailed plot for the condition"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Detailed Analysis: T={temperature}°C, Conc={concentration} mg/mL", 
                 fontsize=16, fontweight="bold")
    
    colors = {"R1":"blue", "R2":"red", "R3":"green"}
    markers = {"R1":"o", "R2":"s", "R3":"^"}
    
    for rxn in ["R1","R2","R3"]:
        subset = reaction_data[rxn]
        if not subset.empty:
            # Conversion vs Time
            axes[0,0].plot(subset["Time_h"], subset["Conversion_pct"], 
                           label=rxn, color=colors[rxn], marker=markers[rxn], 
                           linewidth=2, markersize=6)
            
            # Yield vs Time
            axes[0,1].plot(subset["Time_h"], subset["Yield_pct"], 
                           label=rxn, color=colors[rxn], marker=markers[rxn], 
                           linewidth=2, markersize=6)
            
            # Selectivity vs Time (handle NaN values)
            sel_data = subset["Selectivity_pct"].fillna(0)
            axes[1,0].plot(subset["Time_h"], sel_data, 
                           label=rxn, color=colors[rxn], marker=markers[rxn], 
                           linewidth=2, markersize=6)
            
            # Concentrations vs Time
            axes[1,1].plot(subset["Time_h"], subset["A_mgml"], 
                           label=f"A ({rxn})", linestyle="--", color=colors[rxn], linewidth=2)
            axes[1,1].plot(subset["Time_h"], subset["B_mgml"], 
                           label=f"B ({rxn})", linestyle="-", color=colors[rxn], linewidth=2)
            axes[1,1].plot(subset["Time_h"], subset["I_mgml"], 
                           label=f"I ({rxn})", linestyle=":", color=colors[rxn], linewidth=2)
    
    # Formatting
    titles = ["Conversion vs Time", "Yield vs Time", "Selectivity vs Time", "Concentrations vs Time"]
    ylabels = ["Conversion (%)", "Yield (%)", "Selectivity (%)", "Concentration (mg/mL)"]
    
    for i, (ax, title, ylabel) in enumerate(zip(axes.flatten(), titles, ylabels)):
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if save_plot:
        filename = f"Detailed_Analysis_T{temperature}_C{concentration}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved as: {filename}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def interactive_mode(data):
    """Interactive mode for selecting conditions"""
    
    available_conditions = get_available_conditions(data)
    
    print(f"\n🔍 INTERACTIVE ANALYSIS MODE")
    print(f"{'='*50}")
    print(f"Available Temperature-Concentration combinations:")
    print(f"{'No.':<4} {'Temperature (°C)':<15} {'Concentration (mg/mL)':<20}")
    print("-" * 45)
    
    for i, (temp, conc) in enumerate(available_conditions, 1):
        print(f"{i:<4} {temp:<15} {conc:<20}")
    
    while True:
        try:
            print(f"\nOptions:")
            print(f"1. Enter condition number (1-{len(available_conditions)})")
            print(f"2. Enter custom T,C (e.g., '50,100')")
            print(f"3. Type 'quit' to exit")
            
            user_input = input("\nYour choice: ").strip()
            
            if user_input.lower() in ['quit', 'q', 'exit']:
                print("👋 Goodbye!")
                break
            
            if ',' in user_input:
                # Custom T,C input
                temp_str, conc_str = user_input.split(',')
                temperature = float(temp_str.strip())
                concentration = float(conc_str.strip())
            else:
                # Condition number
                choice = int(user_input)
                if 1 <= choice <= len(available_conditions):
                    temperature, concentration = available_conditions[choice-1]
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(available_conditions)}")
                    continue
            
            # Analyze the selected condition
            analyze_condition(data, temperature, concentration, save_plot=True, show_plot=False)
            
            continue_choice = input("\n🔄 Analyze another condition? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                break
                
        except (ValueError, IndexError):
            print("❌ Invalid input. Please try again.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def main():
    """Main function"""
    print("🧪 DYNAMIC REACTION ANALYSIS TOOL")
    print("="*50)
    
    # Load data
    data = load_reaction_data()
    if data is None:
        print("❌ Failed to load data. Please check your Excel files.")
        return
    
    # Check command line arguments
    if len(sys.argv) == 3:
        try:
            temperature = float(sys.argv[1])
            concentration = float(sys.argv[2])
            print(f"\n📋 Command line mode: T={temperature}°C, C={concentration}mg/mL")
            analyze_condition(data, temperature, concentration)
        except ValueError:
            print("❌ Invalid command line arguments. Use: python script.py <temperature> <concentration>")
    else:
        # Interactive mode
        interactive_mode(data)

if __name__ == "__main__":
    main()
