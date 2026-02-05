import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_PATH = "./polymarket_srs_predictions/analysis_with_srs.csv"

def prepare_data():
    df = pd.read_csv(INPUT_PATH)

    # 1. Parse Dates & Duration
    df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', errors='coerce')
    df['resolved_at'] = pd.to_datetime(df['resolved_at'], format='mixed', errors='coerce')
    df['duration_days'] = (df['resolved_at'] - df['created_at']).dt.days

    # 2. Filter Valid Data
    # We need SRS, Volume, Final Price, and Duration > 0
    df = df.dropna(subset=['predicted_srs', 'volume', 'final_price']).copy()
    df = df[df['duration_days'] > 0]

    # 3. Create "Distance from Certainty" (Convergence Metric)
    # 0.0 = Perfect Certainty (Ends at $0 or $1)
    # 0.5 = Max Confusion (Ends at $0.50)
    df['convergence_gap'] = np.minimum(df['final_price'], 1 - df['final_price'])

    # 4. Log Transforms (for Economic Normality)
    df['log_volume'] = np.log(df['volume'] + 1)
    df['log_duration'] = np.log(df['duration_days'] + 1)

    # 5. Winsorization at 1st and 99th percentiles
    # As described in Section 4.4 of the paper
    df['predicted_srs'] = winsorize(df['predicted_srs'], limits=[0.01, 0.01])
    df['log_volume'] = winsorize(df['log_volume'], limits=[0.01, 0.01])
    df['log_duration'] = winsorize(df['log_duration'], limits=[0.01, 0.01])
    df['convergence_gap'] = winsorize(df['convergence_gap'], limits=[0.01, 0.01])

    print(f"Data Loaded: {len(df)} markets ready for testing.")
    print(f"Winsorization applied at 1st and 99th percentiles.")
    return df

def run_liquidity_test(df):
    print("\n" + "="*60)
    print("TEST 1: LIQUIDITY AVOIDANCE (Volume)")
    print("="*60)
    print("Hypothesis: High SRS -> Low Volume (Negative Coefficient)")

    # Control for Duration (longer markets naturally have more volume)
    model = smf.ols("log_volume ~ predicted_srs + log_duration", data=df).fit()

    print(model.summary().tables[1])

    coef = model.params['predicted_srs']
    pval = model.pvalues['predicted_srs']

    if pval < 0.05:
        print(f"\n✅ SIGNIFICANT FINDING! SRS impacts Volume (p={pval:.4f})")
        print(f"   Direction: {'POSITIVE (More Volume)' if coef > 0 else 'NEGATIVE (Less Volume)'}")
    else:
        print(f"\n❌ Insignificant. SRS does not predict Volume (p={pval:.4f})")

    return model

def run_convergence_test(df):
    print("\n" + "="*60)
    print("TEST 2: CONVERGENCE FAILURE (Price Uncertainty)")
    print("="*60)
    print("Hypothesis: High SRS -> High Gap from 0/1 (Positive Coefficient)")

    # Control for Volume (illiquid markets might just be noisy)
    model = smf.ols("convergence_gap ~ predicted_srs + log_volume", data=df).fit()

    print(model.summary().tables[1])

    coef = model.params['predicted_srs']
    pval = model.pvalues['predicted_srs']

    if pval < 0.05:
        print(f"\n✅ SIGNIFICANT FINDING! SRS impacts Convergence (p={pval:.4f})")
        print(f"   Direction: {'POSITIVE (Worse Convergence)' if coef > 0 else 'NEGATIVE (Better Convergence)'}")
    else:
        print(f"\n❌ Insignificant. SRS does not predict Convergence (p={pval:.4f})")

    return model

def plot_convergence(df):
    plt.figure(figsize=(10, 6))
    sns.regplot(x='predicted_srs', y='convergence_gap', data=df,
                scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
    plt.title('Does Semantic Risk prevent Price Convergence?', fontsize=14)
    plt.xlabel('Semantic Risk Score (SRS)', fontsize=12)
    plt.ylabel('Distance from Certainty (0=Clear, 0.5=Confused)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'convergence_plot.png'")

def print_paper_tables(liquidity_model, convergence_model):
    """Print results formatted for the paper."""
    print("\n" + "="*60)
    print("FORMATTED RESULTS FOR PAPER")
    print("="*60)
    
    print("\nTable 3: Liquidity Regression Results")
    print("-"*60)
    print(f"{'Variable':<20} {'Coef':>10} {'Std.Err':>10} {'t':>10} {'p-value':>10}")
    print("-"*60)
    for var in liquidity_model.params.index:
        print(f"{var:<20} {liquidity_model.params[var]:>10.3f} {liquidity_model.bse[var]:>10.3f} {liquidity_model.tvalues[var]:>10.2f} {liquidity_model.pvalues[var]:>10.4f}")
    print(f"\nN = {int(liquidity_model.nobs)}, R² = {liquidity_model.rsquared:.3f}")
    
    print("\n\nTable 4: Convergence Regression Results")
    print("-"*60)
    print(f"{'Variable':<20} {'Coef':>10} {'Std.Err':>10} {'t':>10} {'p-value':>10}")
    print("-"*60)
    for var in convergence_model.params.index:
        print(f"{var:<20} {convergence_model.params[var]:>10.3f} {convergence_model.bse[var]:>10.3f} {convergence_model.tvalues[var]:>10.2f} {convergence_model.pvalues[var]:>10.4f}")
    print(f"\nN = {int(convergence_model.nobs)}, R² = {convergence_model.rsquared:.3f}")

# Run it
if __name__ == "__main__":
    df = prepare_data()
    liquidity_model = run_liquidity_test(df)
    convergence_model = run_convergence_test(df)
    plot_convergence(df)
    print_paper_tables(liquidity_model, convergence_model)
