import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

def plot_performance_matrix(reaction_name, input_file='Q1 - Rxn_Analysis.xlsx'):
    """
    Generates a heatmap matrix showing impurity, yield, and conversion
    for a specific reaction across all temperature and concentration pairs.

    Args:
        reaction_name (str): The name of the reaction to plot (e.g., 'R1').
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

    # Create condition labels
    reaction_df['condition_label'] = (reaction_df['temperature'].astype(str) + '°C, ' + 
                                     reaction_df['concentration'].astype(str) + ' mg/mL')

    # Sort by temperature and concentration for consistent ordering
    reaction_df = reaction_df.sort_values(['temperature', 'concentration'])

    # Create the matrix data
    matrix_data = []
    condition_labels = []
    
    for _, row in reaction_df.iterrows():
        condition_labels.append(row['condition_label'])
        matrix_data.append([
            row['impurity_formation'],  # Impurity (%)
            row['yield']               # Yield (%)
        ])

    # Convert to numpy array for heatmap
    matrix_array = np.array(matrix_data)
    
    # Create DataFrame for seaborn heatmap
    heatmap_df = pd.DataFrame(
        matrix_array,
        index=condition_labels,
        columns=['Impurity (%)', 'Yield (%)']
    )

    # --- Plotting ---
    # Set plot style and font parameters for readability
    plt.rcParams.update({
        'font.size': 28,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })

    # Create figure
    plt.figure(figsize=(12, 16))

    # Create custom colormap - similar to the original but more appropriate for our metrics
    # Use RdYlGn_r (Red-Yellow-Green reversed) for better interpretation
    # Higher values will be green (good for yield/conversion), lower values red (good for impurity)
    
    # Create the heatmap
    ax = sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Performance Value'},
        linewidths=2,
        linecolor='white',
        annot_kws={'fontsize': 24, 'fontweight': 'bold'},
        square=False
    )

    # Customize the plot
    plt.title(f'PERFORMANCE MATRIX FOR {reaction_name}\nACROSS ALL CONDITIONS', 
              fontsize=24, fontweight='bold', pad=20)
    
    # Customize axis labels
    ax.set_xlabel('Performance Metrics', fontsize=32, fontweight='bold', labelpad=15)
    ax.set_ylabel('Temperature °C, Concentration mg/mL', fontsize=32, fontweight='bold', labelpad=15)
    
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0, fontsize=28, fontweight='bold')
    plt.xticks(rotation=0, fontsize=32, fontweight='bold')
    
    # Adjust colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=26, width=2)
    cbar.set_label('Performance Value', fontsize=28, fontweight='bold', labelpad=20)

    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_filename = f'performance_matrix_heatmap_{reaction_name}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Performance matrix heatmap saved as '{output_filename}'")
    plt.show()


def main():
    """
    Main function to parse command-line arguments and run the plotting function.
    """
    parser = argparse.ArgumentParser(
        description='Generate a performance matrix plot for a specific reaction showing impurity, yield, and conversion.'
    )
    parser.add_argument(
        '--reaction',
        type=str,
        required=True,
        choices=['R1', 'R2', 'R3'],
        help='The name of the reaction to analyze (e.g., "R1").'
    )
    args = parser.parse_args()
    plot_performance_matrix(args.reaction)


if __name__ == '__main__':
    main()
