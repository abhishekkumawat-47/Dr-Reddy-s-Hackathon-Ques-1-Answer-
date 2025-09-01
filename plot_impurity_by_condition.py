
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_impurity_by_condition(reaction_name, input_file='Q1 - Rxn_Analysis.xlsx'):
    """
    Generates a bar plot of impurity formation for a specific reaction
    across all temperature and concentration pairs.

    Args:
        reaction_name (str): The name of the reaction to plot (e.g., 'Reaction-1').
        input_file (str): The path to the Excel file containing the analysis data.
    """
    # --- Data Loading and Preparation ---
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    try:
        # Load the data from the "Condition_by_Condition" sheet
        df = pd.read_excel(input_file, sheet_name='Condition_by_Condition')
    except ValueError:
        print(f"Error: Sheet 'Condition_by_Condition' not found in '{input_file}'.")
        return

    # Filter data for the specified reaction
    reaction_df = df[df['reaction'] == reaction_name].copy()

    if reaction_df.empty:
        print(f"Error: No data found for '{reaction_name}'. Please check the reaction name.")
        return

    # Create a descriptive label for each condition for the x-axis
    reaction_df['Condition'] = reaction_df['temperature'].astype(str) + '°C / ' + \
                               reaction_df['concentration'].astype(str) + ' mg/mL'

    # --- Plotting ---
    # Define colors to match the executive summary plots
    color_map = {
        'R1': 'seagreen',
        'R2': 'chocolate', 
        'R3': 'firebrick'
    }
    plot_color = color_map.get(reaction_name, 'royalblue')

    # Set plot style and font parameters for readability
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })

    # Create a larger figure
    plt.figure(figsize=(18, 10))

    # Create the bar plot
    ax = sns.barplot(
        x='Condition',
        y='impurity_formation',
        data=reaction_df,
        color=plot_color
    )

    # --- Labels and Titles ---
    plt.title(f'Impurity Formation Across All Conditions for {reaction_name}', fontsize=20, fontweight='bold')
    plt.xlabel('Temperature / Concentration Pair', fontsize=16, fontweight='bold')
    plt.ylabel('Impurity Formation (%)', fontsize=16, fontweight='bold')

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right')

    # Add value labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold')

    # Adjust layout and save the plot
    plt.tight_layout()
    output_filename = f'impurity_plot_{reaction_name}.png'
    plt.savefig(output_filename)
    print(f"Plot saved as '{output_filename}'")
    plt.show()


def main():
    """
    Main function to parse command-line arguments and run the plotting function.
    """
    parser = argparse.ArgumentParser(
        description='Generate a plot of impurity formation for a specific reaction.'
    )
    parser.add_argument(
        '--reaction',
        type=str,
        required=True,
        choices=['R1', 'R2', 'R3'],
        help='The name of the reaction to analyze (e.g., "R1").'
    )
    args = parser.parse_args()
    plot_impurity_by_condition(args.reaction)


if __name__ == '__main__':
    main()
