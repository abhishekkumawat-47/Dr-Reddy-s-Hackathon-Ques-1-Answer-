import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("📊 CREATING COMPREHENSIVE VISUALIZATION OF RESULTS")
print("="*60)

# Load the saved results
df_conditions = pd.read_excel('Comprehensive_Reaction_Analysis.xlsx', sheet_name='Condition_by_Condition')
df_overall = pd.read_excel('Comprehensive_Reaction_Analysis.xlsx', sheet_name='Overall_Rankings')

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))

# 1. Overall Score Comparison
ax1 = plt.subplot(3, 4, 1)
reactions = df_overall['reaction'].tolist()
scores = df_overall['overall_score'].tolist()
colors = ['#2E8B57', '#CD853F', '#B22222']  # Green, Orange, Red

bars = plt.bar(reactions, scores, color=colors)
plt.title('Overall Performance Scores\n(Higher = Better)', fontweight='bold', fontsize=12)
plt.ylabel('Score')
plt.ylim(0, 4)

# Add score values on bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Selectivity Comparison
ax2 = plt.subplot(3, 4, 2)
selectivities = df_overall['avg_selectivity'].tolist()
plt.bar(reactions, selectivities, color=colors)
plt.title('Average Selectivity\n(Priority #1)', fontweight='bold', fontsize=12)
plt.ylabel('Selectivity (%)')
plt.ylim(0, 100)

# 3. Temperature & Concentration Sensitivity
ax3 = plt.subplot(3, 4, 3)
temp_sens = df_overall['avg_temp_sensitivity'].tolist()
conc_sens = df_overall['avg_conc_sensitivity'].tolist()

x = np.arange(len(reactions))
width = 0.35

plt.bar(x - width/2, temp_sens, width, label='Temp Sensitivity', color='lightcoral')
plt.bar(x + width/2, conc_sens, width, label='Conc Sensitivity', color='lightskyblue')
plt.title('Sensitivity Analysis\n(Priority #2 - Lower = Better)', fontweight='bold', fontsize=12)
plt.ylabel('Sensitivity')
plt.xticks(x, reactions)
plt.legend()

# 4. Rate Constants
ax4 = plt.subplot(3, 4, 4)
rates = df_overall['avg_rate'].tolist()
plt.bar(reactions, rates, color=colors)
plt.title('Average Rate Constants\n(Priority #3)', fontweight='bold', fontsize=12)
plt.ylabel('Rate (%/h)')

# 5. Impurity Formation
ax5 = plt.subplot(3, 4, 5)
impurities = df_overall['avg_impurities'].tolist()
plt.bar(reactions, impurities, color=colors)
plt.title('Average Impurity Formation\n(Priority #4 - Lower = Better)', fontweight='bold', fontsize=12)
plt.ylabel('Impurities (%)')

# 6. Conversion & Yield
ax6 = plt.subplot(3, 4, 6)
conversions = df_overall['avg_conversion'].tolist()
yields = df_overall['avg_yield'].tolist()

x = np.arange(len(reactions))
plt.bar(x - width/2, conversions, width, label='Conversion', color='lightgreen')
plt.bar(x + width/2, yields, width, label='Yield', color='gold')
plt.title('Conversion & Yield\n(Priority #5 & #6)', fontweight='bold', fontsize=12)
plt.ylabel('Percentage (%)')
plt.xticks(x, reactions)
plt.legend()

# 7. Wins per Condition
ax7 = plt.subplot(3, 4, 7)
wins = df_overall['first_place_count'].tolist()
total_conditions = df_overall['conditions_analyzed'].iloc[0]
plt.bar(reactions, wins, color=colors)
plt.title(f'Wins per Condition\n(Out of {total_conditions} conditions)', fontweight='bold', fontsize=12)
plt.ylabel('Number of Wins')
plt.ylim(0, total_conditions)

# 8. Consistency Score
ax8 = plt.subplot(3, 4, 8)
consistency = df_overall['consistency'].tolist()
plt.bar(reactions, consistency, color=colors)
plt.title('Performance Consistency\n(Higher = Better)', fontweight='bold', fontsize=12)
plt.ylabel('Consistency Score')
plt.ylim(0, 1)

# 9. Score Distribution Across Conditions (Heatmap)
ax9 = plt.subplot(3, 4, (9, 12))
# Create pivot table for heatmap
pivot_data = df_conditions.pivot_table(
    index=['temperature', 'concentration'], 
    columns='reaction', 
    values='score_at_condition'
)

# Create condition labels
condition_labels = [f"{int(temp)}°C, {int(conc)}mg/mL" 
                   for temp, conc in zip(df_conditions['temperature'].unique(), 
                                       df_conditions['concentration'].unique())]

# Create a proper heatmap
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
            cbar_kws={'label': 'Score'}, ax=ax9)
ax9.set_title('Performance Score Matrix\nAcross All Conditions', fontweight='bold', fontsize=14)
ax9.set_xlabel('Reaction')
ax9.set_ylabel('Condition (Temp, Conc)')

plt.tight_layout()
plt.savefig('Comprehensive_Reaction_Analysis_Plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a summary table
print("\n📋 EXECUTIVE SUMMARY TABLE")
print("="*80)

summary_data = []
for _, row in df_overall.iterrows():
    summary_data.append({
        'Reaction': row['reaction'],
        'Overall Score': f"{row['overall_score']:.3f}",
        'Rank': f"#{df_overall.index[df_overall['reaction'] == row['reaction']].tolist()[0] + 1}",
        'Wins/Total': f"{row['first_place_count']}/{row['conditions_analyzed']}",
        'Selectivity': f"{row['avg_selectivity']:.1f}%",
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
print(f"• R1 has the BEST selectivity (96.6%) - your #1 priority")
print(f"• R1 has LOWEST sensitivity to temp/conc changes - your #2 priority")  
print(f"• R1 has FASTEST rate (20%/h) - your #3 priority")
print(f"• R1 produces LEAST impurities (3.4%) - your #4 priority")
print(f"• R2 has MAJOR selectivity issues (10.7% avg) - high impurity formation")
print(f"• R3 is decent but SLOW and more sensitive to conditions")

print(f"\n💡 BUSINESS RECOMMENDATION:")
print(f"IMPLEMENT R1 as your primary reaction pathway!")
print(f"It consistently outperforms across ALL priority criteria and conditions.")

print(f"\n📊 Files Generated:")
print(f"• Comprehensive_Reaction_Analysis.xlsx - Detailed data")
print(f"• Comprehensive_Reaction_Analysis_Plots.png - Visualizations")
