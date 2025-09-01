import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("📊 CREATING IMPROVED COMPREHENSIVE VISUALIZATION")
print("="*60)

# Load the saved results
df_conditions = pd.read_excel('Q1 - Rxn_Analysis.xlsx', sheet_name='Condition_by_Condition')
df_overall = pd.read_excel('Q1 - Rxn_Analysis.xlsx', sheet_name='Overall_Rankings')

# Ensure consistent ordering: R1, R2, R3
reaction_order = ['R1', 'R2', 'R3']
df_overall = df_overall.set_index('reaction').reindex(reaction_order).reset_index()

# Set global font sizes for enhanced readability
plt.rcParams.update({
    'font.size': 32,           # Increased base font size
    'axes.titlesize': 32,      # Much larger subplot titles
    'axes.labelsize': 26,      # Much larger axis labels
    'xtick.labelsize': 28,     # Much larger X-axis tick labels
    'ytick.labelsize': 28,     # Much larger Y-axis tick labels
    'legend.fontsize': 28,     # Much larger legend
    'figure.titlesize': 44,    # Much larger main title
    'lines.linewidth': 4.0,    # Even thicker lines
    'grid.linewidth': 1.6,     # Thicker grid lines
    'axes.linewidth': 2.2,     # Thicker axes border
    'xtick.major.size': 20,    # Larger x-axis tick marks
    'ytick.major.size': 20,    # Larger y-axis tick marks
    'xtick.major.width': 2.5,  # Thicker x-axis tick marks
    'ytick.major.width': 2.5,  # Thicker y-axis tick marks
    'font.weight': 'bold'      # Make all fonts bold by default
})

# ========================================
# IMAGE 1: ALL 6 COMPARISON PLOTS
# ========================================

fig1 = plt.figure(figsize=(32, 18))  # Increased size for better readability
fig1.suptitle('COMPREHENSIVE REACTION PERFORMANCE COMPARISON', fontsize=36, fontweight='bold', y=0.95)

# Define consistent colors for R1, R2, R3
colors = ['#2E8B57', '#CD853F', '#B22222']  # Green, Orange, Red
reactions = df_overall['reaction'].tolist()

# 1. Overall Performance Scores (Summary)
ax1 = plt.subplot(2, 3, 1)
scores = df_overall['overall_score'].tolist()
bars = plt.bar(reactions, scores, color=colors, width=0.6, edgecolor='black', linewidth=3)
plt.title('OVERALL PERFORMANCE SCORES\n(Weighted by All Priorities)', fontweight='bold', fontsize=24, pad=25)
plt.ylabel('Overall Score', fontsize=20, fontweight='bold')
plt.ylim(0, max(scores) * 1.2)
plt.grid(True, alpha=0.3, axis='y')

# Add score values on bars with larger font
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=18)

# 2. Priority #1: Impurity Formation (Most Important)
ax2 = plt.subplot(2, 3, 2)
impurities = df_overall['avg_impurities'].tolist()
bars = plt.bar(reactions, impurities, color=colors, width=0.6, edgecolor='black', linewidth=3)
plt.title('PRIORITY #1: Impurity Formation\n(Lower = Better)', fontweight='bold', fontsize=24, pad=25)
plt.ylabel('Impurities (%)', fontsize=20, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add values on bars with larger font
for bar, imp in zip(bars, impurities):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=18)

# 3. Priority #2: Yield (Second Most Important)
ax3 = plt.subplot(2, 3, 3)
yields = df_overall['avg_yield'].tolist()
bars = plt.bar(reactions, yields, color=colors, width=0.6, edgecolor='black', linewidth=3)
plt.title('PRIORITY #2: Yield\n(Higher = Better)', fontweight='bold', fontsize=24, pad=25)
plt.ylabel('Yield (%)', fontsize=20, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, yld in zip(bars, yields):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{yld:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=18)

# 4. Priority #3: Rate Constants (Third Priority)
ax4 = plt.subplot(2, 3, 4)
rates = df_overall['avg_rate'].tolist()
bars = plt.bar(reactions, rates, color=colors, width=0.6, edgecolor='black', linewidth=3)
plt.title('PRIORITY #3: Average Rate Constants\n(Higher = Better)', fontweight='bold', fontsize=24, pad=25)
plt.ylabel('Rate (%/h)', fontsize=20, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, rate in zip(bars, rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{rate:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=18)

# 5. Priority #4: Sensitivity Analysis (Temperature & Concentration)
ax5 = plt.subplot(2, 3, 5)
temp_sens = df_overall['avg_temp_sensitivity'].tolist()
conc_sens = df_overall['avg_conc_sensitivity'].tolist()

x = np.arange(len(reactions))
width = 0.35

# Use consistent colors for R1, R2, R3 with different patterns
temp_colors = ['#2E8B57', '#CD853F', '#B22222']  # Same as main colors
conc_colors = ['#1C5D3A', '#A0692B', '#8B1A1A']  # Darker shades

bars1 = plt.bar(x - width/2, temp_sens, width, label='Temp Sensitivity', color=temp_colors, edgecolor='black', linewidth=2)
bars2 = plt.bar(x + width/2, conc_sens, width, label='Conc Sensitivity', color=conc_colors, edgecolor='black', linewidth=2)
plt.title('PRIORITY #4: Sensitivity Analysis\n(Lower = Better)', fontweight='bold', fontsize=24, pad=25)
plt.ylabel('Sensitivity', fontsize=20, fontweight='bold')
plt.xticks(x, reactions)
plt.legend(fontsize=18, loc='upper right')
plt.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars1, temp_sens):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=16)
for bar, val in zip(bars2, conc_sens):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=16)

# 6. Wins Distribution (Condition Analysis)
ax6 = plt.subplot(2, 3, 6)
wins = df_overall['first_place_count'].tolist()
total_conditions = df_overall['conditions_analyzed'].tolist()
win_percentages = [(w/t)*100 for w, t in zip(wins, total_conditions)]

bars = plt.bar(reactions, win_percentages, color=colors, width=0.6, edgecolor='black', linewidth=3)
plt.title('WINNING PERCENTAGE\n(Conditions Won)', fontweight='bold', fontsize=24, pad=25)
plt.ylabel('Win Percentage (%)', fontsize=20, fontweight='bold')
plt.ylim(0, 105)
plt.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, win_pct, wins_count, total in zip(bars, win_percentages, wins, total_conditions):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{win_pct:.0f}%\n({wins_count}/{total})', ha='center', va='bottom', fontweight='bold', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('Reaction_Performance_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ========================================
# IMAGE 2: PERFORMANCE SCORE MATRIX
# ========================================

fig2 = plt.figure(figsize=(24, 14))  # Much larger for better readability
fig2.suptitle('PERFORMANCE SCORE MATRIX ACROSS ALL CONDITIONS', fontsize=32, fontweight='bold', y=0.95)

# Create pivot table for heatmap
pivot_data = df_conditions.pivot_table(
    index=['temperature', 'concentration'], 
    columns='reaction', 
    values='score_at_condition'
)

# Ensure R1, R2, R3 column order
pivot_data = pivot_data.reindex(columns=['R1', 'R2', 'R3'])

# Create the heatmap with much larger fonts and better spacing
ax = plt.subplot(1, 1, 1)
heatmap = sns.heatmap(pivot_data, 
                     annot=True, 
                     fmt='.3f', 
                     cmap='RdYlGn', 
                     cbar_kws={'label': 'Performance Score', 'shrink': 0.6, 'pad': 0.02},
                     annot_kws={'size': 18, 'weight': 'bold'},  # Much larger annotation font
                     linewidths=3,  # Thicker lines
                     linecolor='white',
                     square=False,  # Allow rectangular cells for better spacing
                     xticklabels=True,
                     yticklabels=True)

# Customize the heatmap with better spacing and larger fonts
heatmap.set_xlabel('Reaction', fontsize=24, fontweight='bold', labelpad=20)
heatmap.set_ylabel('Condition (Temperature °C, Concentration mg/mL)', fontsize=24, fontweight='bold', labelpad=20)

# Improve tick labels with much larger fonts
heatmap.tick_params(axis='x', labelsize=22, pad=10)
heatmap.tick_params(axis='y', labelsize=18, pad=10)

# Format y-axis labels for better readability
y_labels = [f"{int(temp)}°C, {int(conc)}mg/mL" for temp, conc in pivot_data.index]
heatmap.set_yticklabels(y_labels, rotation=0, ha='right', fontweight='bold')

# Rotate x-axis labels for better readability
plt.setp(heatmap.get_xticklabels(), rotation=0, ha='center', fontweight='bold')

# Add colorbar label with much larger font
cbar = heatmap.collections[0].colorbar
cbar.set_label('Performance Score', fontsize=20, fontweight='bold', labelpad=25)
cbar.ax.tick_params(labelsize=18)

# Add more padding around the plot
plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('Performance_Score_Matrix.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# ========================================
# SUMMARY TABLE
# ========================================

print("\n📋 EXECUTIVE SUMMARY TABLE")
print("="*80)

summary_data = []
for _, row in df_overall.iterrows():
    summary_data.append({
        'Reaction': row['reaction'],
        'Overall Score': f"{row['overall_score']:.3f}",
        'Rank': f"#{df_overall.index[df_overall['reaction'] == row['reaction']].tolist()[0] + 1}",
        'Wins/Total': f"{row['first_place_count']}/{row['conditions_analyzed']}",
        'Yield': f"{row['avg_yield']:.1f}%", 
        'Rate': f"{row['avg_rate']:.1f}%/h",
        'Impurities': f"{row['avg_impurities']:.1f}%",
        'Temp Sens': f"{row['avg_temp_sensitivity']:.3f}",
        'Conc Sens': f"{row['avg_conc_sensitivity']:.3f}"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\n🎯 KEY FINDINGS:")
print(f"• R1 DOMINATES: Wins ALL 15 conditions tested")
print(f"• R1 produces LEAST impurities (3.4%) - your #1 priority")
print(f"• R1 has HIGHEST yield (96.4%) - your #2 priority")
print(f"• R1 has FASTEST rate (20%/h) - your #3 priority")
print(f"• R1 has LOWEST sensitivity to temp/conc changes - your #4 priority")
print(f"• R2 has MAJOR impurity issues (89.2% avg) - fails priority #1")
print(f"• R3 is decent but SLOW and produces more impurities")

print(f"\n💡 BUSINESS RECOMMENDATION:")
print(f"IMPLEMENT R1 as your primary reaction pathway!")
print(f"It consistently outperforms across ALL priority criteria and conditions.")

print(f"\n📊 Files Generated:")
print(f"• Reaction_Performance_Comparison.png - Comparison plots")
print(f"• Performance_Score_Matrix.png - Heatmap visualization")
print(f"• Q1 - Rxn_Analysis.xlsx - Detailed data")

# Reset font settings
plt.rcParams.update(plt.rcParamsDefault)
