import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

print("🏆 COMPREHENSIVE REACTION RANKING SYSTEM")
print("="*70)

class ReactionRanker:
    def __init__(self):
        # Priority weights based on your NEW requirements
        self.priority_weights = {
            'impurity_formation': 1.0,    # Priority 1 - Most important (minimization)
            'yield': 0.9,                # Priority 2 - Maximize yield = more product, less waste
            'conversion': 0.85,          # Priority 3 - Maximize substrate utilization
            'selectivity': 0.7,          # Priority 4 - Still important for quality and downstream costs
            'rate_constant': 0.55,       # Priority 5 - Throughput, but secondary to purity/yield
            'temp_sensitivity': 0.4,     # Priority 6 - Stability
            'conc_sensitivity': 0.4,     # Priority 6 - Robustness
            'others': 0.2                # Priority 7
        }
        
        self.data = {}
        self.results = []
        
    def load_data(self):
        """Load all reaction data"""
        files = {
            "R1": "Reaction-1.xlsx",
            "R2": "Reaction-2.xlsx", 
            "R3": "Reaction-3.xlsx"
        }
        
        print("📊 Loading reaction data...")
        for rxn, file in files.items():
            self.data[rxn] = pd.read_excel(file, sheet_name="Calculated")
            print(f"  ✓ {rxn}: {len(self.data[rxn])} data points")
            
    def get_conditions(self):
        """Get all unique temperature-concentration pairs"""
        conditions = set()
        for rxn_data in self.data.values():
            for _, row in rxn_data.iterrows():
                conditions.add((row['Temp_C'], row['A0_mgml']))
        return sorted(list(conditions))
    
    def calculate_sensitivity_metrics(self, rxn):
        """Calculate temperature and concentration sensitivity"""
        df = self.data[rxn]
        
        # Temperature sensitivity: how much selectivity changes with temperature
        temp_effects = []
        for conc in df['A0_mgml'].unique():
            conc_data = df[df['A0_mgml'] == conc]
            if len(conc_data['Temp_C'].unique()) > 1:
                # Get final selectivity for each temperature
                final_sel = []
                temps = []
                for temp in sorted(conc_data['Temp_C'].unique()):
                    temp_data = conc_data[conc_data['Temp_C'] == temp]
                    if not temp_data.empty:
                        final_sel.append(temp_data['Selectivity_pct'].iloc[-1])
                        temps.append(temp)
                
                if len(final_sel) > 1:
                    # Calculate coefficient of variation (std/mean)
                    temp_effects.append(np.std(final_sel) / (np.mean(final_sel) + 1e-6))
        
        temp_sensitivity = np.mean(temp_effects) if temp_effects else 0
        
        # Concentration sensitivity: how much selectivity changes with concentration
        conc_effects = []
        for temp in df['Temp_C'].unique():
            temp_data = df[df['Temp_C'] == temp]
            if len(temp_data['A0_mgml'].unique()) > 1:
                # Get final selectivity for each concentration
                final_sel = []
                concs = []
                for conc in sorted(temp_data['A0_mgml'].unique()):
                    conc_data = temp_data[temp_data['A0_mgml'] == conc]
                    if not conc_data.empty:
                        final_sel.append(conc_data['Selectivity_pct'].iloc[-1])
                        concs.append(conc)
                
                if len(final_sel) > 1:
                    # Calculate coefficient of variation
                    conc_effects.append(np.std(final_sel) / (np.mean(final_sel) + 1e-6))
        
        conc_sensitivity = np.mean(conc_effects) if conc_effects else 0
        
        return temp_sensitivity, conc_sensitivity
    
    def analyze_condition(self, temp, conc):
        """Analyze all reactions at a specific temperature and concentration"""
        print(f"\n🔬 Analyzing T={temp}°C, C={conc}mg/mL")
        print("-" * 50)
        
        condition_results = {}
        
        for rxn in ["R1", "R2", "R3"]:
            df = self.data[rxn]
            subset = df[(df["Temp_C"] == temp) & (df["A0_mgml"] == conc)]
            
            if subset.empty:
                print(f"  ⚠️  No data for {rxn}")
                continue
                
            # Extract key metrics
            final_row = subset.iloc[-1]
            initial_row = subset.iloc[0]
            
            # Calculate rate constant (simplified)
            time_span = final_row['Time_h'] - initial_row['Time_h']
            if time_span > 0:
                rate_constant = (final_row['Conversion_pct'] - initial_row['Conversion_pct']) / time_span
            else:
                rate_constant = 0
            
            # Get sensitivity metrics
            temp_sens, conc_sens = self.calculate_sensitivity_metrics(rxn)
            
            metrics = {
                'reaction': rxn,
                'temperature': temp,
                'concentration': conc,
                'selectivity': final_row['Selectivity_pct'] if pd.notna(final_row['Selectivity_pct']) else 0,
                'temp_sensitivity': temp_sens,
                'conc_sensitivity': conc_sens,
                'rate_constant': rate_constant,
                'impurity_formation': final_row['I_mgml'] / conc * 100,  # % impurities formed
                'conversion': final_row['Conversion_pct'],
                'yield': final_row['Yield_pct'],
                'time_to_completion': final_row['Time_h'],
                'final_A': final_row['A_mgml'],
                'final_B': final_row['B_mgml'],
                'final_I': final_row['I_mgml']
            }
            
            condition_results[rxn] = metrics
            
        # Rank reactions for this condition
        ranked = self.rank_reactions_for_condition(condition_results)
        
        # Store results
        for rank, (rxn, score, metrics) in enumerate(ranked, 1):
            result = metrics.copy()
            result['rank_at_condition'] = rank
            result['score_at_condition'] = score
            self.results.append(result)
            
        # Display results for this condition
        print(f"\n📊 Results for T={temp}°C, C={conc}mg/mL:")
        print(f"{'Rank':<4} {'Reaction':<8} {'Score':<6} {'Selectivity':<10} {'Yield':<8} {'Rate':<8} {'Impurities':<10}")
        print("-" * 70)
        for rank, (rxn, score, metrics) in enumerate(ranked, 1):
            print(f"{rank:<4} {rxn:<8} {score:<6.3f} {metrics['selectivity']:<10.1f} {metrics['yield']:<8.1f} {metrics['rate_constant']:<8.1f} {metrics['impurity_formation']:<10.1f}")
            
        return ranked
    
    def rank_reactions_for_condition(self, condition_results):
        """Rank reactions based on priority hierarchy for a specific condition"""
        if not condition_results:
            return []
            
        # Normalize metrics for scoring
        reactions = list(condition_results.keys())
        
        # Extract values for normalization
        selectivities = [condition_results[r]['selectivity'] for r in reactions]
        temp_sens = [condition_results[r]['temp_sensitivity'] for r in reactions]
        conc_sens = [condition_results[r]['conc_sensitivity'] for r in reactions]
        rates = [condition_results[r]['rate_constant'] for r in reactions]
        impurities = [condition_results[r]['impurity_formation'] for r in reactions]
        conversions = [condition_results[r]['conversion'] for r in reactions]
        yields = [condition_results[r]['yield'] for r in reactions]
        
        # Normalize to 0-1 scale
        def normalize(values, invert=False):
            if not values or max(values) == min(values):
                return [0.5] * len(values)
            normalized = [(v - min(values)) / (max(values) - min(values)) for v in values]
            return [1 - n for n in normalized] if invert else normalized
        
        norm_selectivity = normalize(selectivities)
        norm_temp_sens = normalize(temp_sens, invert=True)  # Lower sensitivity is better
        norm_conc_sens = normalize(conc_sens, invert=True)  # Lower sensitivity is better
        norm_rates = normalize(rates)
        norm_impurities = normalize(impurities, invert=True)  # Lower impurities is better
        norm_conversions = normalize(conversions)
        norm_yields = normalize(yields)
        
        # Calculate weighted scores
        scores = []
        for i, rxn in enumerate(reactions):
            score = (
                norm_impurities[i] * self.priority_weights['impurity_formation'] +
                norm_yields[i] * self.priority_weights['yield'] +
                norm_conversions[i] * self.priority_weights['conversion'] +
                norm_selectivity[i] * self.priority_weights['selectivity'] +
                norm_rates[i] * self.priority_weights['rate_constant'] +
                norm_temp_sens[i] * self.priority_weights['temp_sensitivity'] +
                norm_conc_sens[i] * self.priority_weights['conc_sensitivity']
            )
            scores.append((rxn, score, condition_results[rxn]))
            
        # Sort by score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def calculate_overall_rankings(self):
        """Calculate overall rankings across all conditions"""
        print(f"\n🏆 CALCULATING OVERALL RANKINGS")
        print("="*60)
        
        # Group results by reaction
        reaction_scores = defaultdict(list)
        reaction_metrics = defaultdict(list)
        
        for result in self.results:
            rxn = result['reaction']
            reaction_scores[rxn].append(result['score_at_condition'])
            reaction_metrics[rxn].append(result)
        
        # Calculate aggregate metrics for each reaction
        overall_results = []
        
        for rxn in reaction_scores.keys():
            scores = reaction_scores[rxn]
            metrics = reaction_metrics[rxn]
            
            # Calculate aggregated metrics
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            consistency = 1 / (1 + std_score)  # Higher consistency is better
            
            # Average key metrics across all conditions
            avg_selectivity = np.mean([m['selectivity'] for m in metrics])
            avg_yield = np.mean([m['yield'] for m in metrics])
            avg_conversion = np.mean([m['conversion'] for m in metrics])
            avg_impurities = np.mean([m['impurity_formation'] for m in metrics])
            avg_rate = np.mean([m['rate_constant'] for m in metrics])
            avg_temp_sens = np.mean([m['temp_sensitivity'] for m in metrics])
            avg_conc_sens = np.mean([m['conc_sensitivity'] for m in metrics])
            
            # Count how many times this reaction was ranked #1
            first_place_count = sum(1 for m in metrics if m['rank_at_condition'] == 1)
            
            overall_results.append({
                'reaction': rxn,
                'overall_score': avg_score,
                'consistency': consistency,
                'first_place_count': first_place_count,
                'avg_selectivity': avg_selectivity,
                'avg_yield': avg_yield,
                'avg_conversion': avg_conversion,
                'avg_impurities': avg_impurities,
                'avg_rate': avg_rate,
                'avg_temp_sensitivity': avg_temp_sens,
                'avg_conc_sensitivity': avg_conc_sens,
                'conditions_analyzed': len(metrics)
            })
        
        # Sort by overall score
        overall_results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return overall_results
    
    def display_detailed_results(self):
        """Display comprehensive results"""
        overall_results = self.calculate_overall_rankings()
        
        print(f"\n📈 DETAILED CONDITION-BY-CONDITION ANALYSIS")
        print("="*80)
        
        # Create summary table by condition
        conditions = self.get_conditions()
        
        print(f"\n📊 PERFORMANCE MATRIX (Score at each condition):")
        print(f"{'Temp(°C)':<8} {'Conc(mg/mL)':<12} {'R1 Score':<10} {'R2 Score':<10} {'R3 Score':<10} {'Winner':<8}")
        print("-" * 70)
        
        condition_winners = defaultdict(int)
        
        for temp, conc in conditions:
            condition_data = [r for r in self.results if r['temperature'] == temp and r['concentration'] == conc]
            if condition_data:
                condition_data.sort(key=lambda x: x['rank_at_condition'])
                
                scores = {r['reaction']: r['score_at_condition'] for r in condition_data}
                winner = condition_data[0]['reaction']
                condition_winners[winner] += 1
                
                r1_score = scores.get('R1', 0)
                r2_score = scores.get('R2', 0)  
                r3_score = scores.get('R3', 0)
                
                print(f"{temp:<8} {conc:<12} {r1_score:<10.3f} {r2_score:<10.3f} {r3_score:<10.3f} {winner:<8}")
        
        print(f"\n🏅 OVERALL REACTION RANKINGS")
        print("="*60)
        
        for i, result in enumerate(overall_results, 1):
            print(f"\n{i}. {result['reaction']} - Overall Score: {result['overall_score']:.3f}")
            print(f"   🎯 Wins: {result['first_place_count']}/{result['conditions_analyzed']} conditions")
            print(f"   📊 Key Metrics:")
            print(f"      • Selectivity: {result['avg_selectivity']:.1f}%")
            print(f"      • Yield: {result['avg_yield']:.1f}%")
            print(f"      • Conversion: {result['avg_conversion']:.1f}%")
            print(f"      • Impurities: {result['avg_impurities']:.1f}%")
            print(f"      • Rate: {result['avg_rate']:.1f}%/h")
            print(f"      • Temp Sensitivity: {result['avg_temp_sensitivity']:.3f}")
            print(f"      • Conc Sensitivity: {result['avg_conc_sensitivity']:.3f}")
            print(f"      • Consistency: {result['consistency']:.3f}")
        
        print(f"\n🎯 FINAL CONCLUSION")
        print("="*50)
        
        winner = overall_results[0]
        runner_up = overall_results[1] if len(overall_results) > 1 else None
        
        print(f"🥇 RECOMMENDED REACTION: {winner['reaction']}")
        print(f"   Overall Score: {winner['overall_score']:.3f}")
        print(f"   Wins {winner['first_place_count']} out of {winner['conditions_analyzed']} conditions")
        
        print(f"\n💡 WHY {winner['reaction']} IS THE BEST CHOICE:")
        
        # Analyze why this reaction won
        reasons = []
        
        if winner['avg_selectivity'] > 90:
            reasons.append(f"✅ Excellent selectivity ({winner['avg_selectivity']:.1f}%)")
        
        if winner['avg_temp_sensitivity'] < 0.1:
            reasons.append(f"✅ Low temperature sensitivity ({winner['avg_temp_sensitivity']:.3f})")
            
        if winner['avg_conc_sensitivity'] < 0.1:
            reasons.append(f"✅ Low concentration sensitivity ({winner['avg_conc_sensitivity']:.3f})")
            
        if winner['avg_rate'] > 10:
            reasons.append(f"✅ Fast reaction rate ({winner['avg_rate']:.1f}%/h)")
            
        if winner['avg_impurities'] < 10:
            reasons.append(f"✅ Low impurity formation ({winner['avg_impurities']:.1f}%)")
            
        if winner['consistency'] > 0.8:
            reasons.append(f"✅ Consistent performance (consistency: {winner['consistency']:.3f})")
        
        for reason in reasons:
            print(f"   {reason}")
        
        if runner_up:
            print(f"\n🥈 Runner-up: {runner_up['reaction']} (Score: {runner_up['overall_score']:.3f})")
            
            # Compare with winner
            if runner_up['avg_selectivity'] > winner['avg_selectivity']:
                print(f"   • Better selectivity than {winner['reaction']}")
            if runner_up['avg_rate'] > winner['avg_rate']:
                print(f"   • Faster than {winner['reaction']}")
            
        return overall_results

def main():
    ranker = ReactionRanker()
    ranker.load_data()
    
    conditions = ranker.get_conditions()
    print(f"\n🔍 Found {len(conditions)} temperature-concentration conditions to analyze")
    
    # Analyze each condition
    for temp, conc in conditions:
        ranker.analyze_condition(temp, conc)
    
    # Display comprehensive results
    overall_results = ranker.display_detailed_results()
    
    # Save detailed results
    results_df = pd.DataFrame(ranker.results)
    overall_df = pd.DataFrame(overall_results)
    
    with pd.ExcelWriter('Comprehensive_Reaction_Analysis.xlsx', engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Condition_by_Condition', index=False)
        overall_df.to_excel(writer, sheet_name='Overall_Rankings', index=False)
    
    print(f"\n💾 Results saved to: Comprehensive_Reaction_Analysis.xlsx")
    
    return overall_results

if __name__ == "__main__":
    results = main()
