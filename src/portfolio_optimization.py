# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_PATH = "/content/polymarket_srs_predictions/analysis_with_srs.csv"  # Change this to your file path

SRS_THRESHOLD = 0.15      # Default clarity threshold (will test multiple)
GAMMA = 1                  # Risk aversion parameter (1 = linear penalty)
MIN_VIABLE_VOLUME = 5000   # Definition of "dead" market

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(path):
    """Load and prepare the dataset."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    df = pd.read_csv(path)
    
    # Parse dates
    df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', errors='coerce')
    df['resolved_at'] = pd.to_datetime(df['resolved_at'], format='mixed', errors='coerce')
    df['duration_days'] = (df['resolved_at'] - df['created_at']).dt.days
    
    # Filter for valid data
    df = df.dropna(subset=['predicted_srs', 'volume']).copy()
    
    print(f"Total markets loaded: {len(df)}")
    print(f"SRS range: [{df['predicted_srs'].min():.3f}, {df['predicted_srs'].max():.3f}]")
    print(f"Volume range: [${df['volume'].min():,.0f}, ${df['volume'].max():,.0f}]")
    print()
    
    return df

# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================

def naive_portfolio(df):
    """Equal weight across all markets."""
    df = df.copy()
    df['weight'] = 1 / len(df)
    return df

def clarity_portfolio(df, threshold=0.15, gamma=1):
    """Clarity-weighted portfolio with SRS threshold."""
    df = df.copy()
    df['weight'] = 0.0  # Initialize as float
    
    # Filter by threshold
    mask = df['predicted_srs'] <= threshold
    eligible = df[mask].copy()
    
    if len(eligible) == 0:
        print(f"WARNING: No markets pass threshold {threshold}")
        return df
    
    # Calculate clarity score and weights
    clarity_scores = (1 - eligible['predicted_srs']) ** gamma
    total_clarity = clarity_scores.sum()
    weights = clarity_scores / total_clarity
    
    # Assign weights
    df.loc[mask, 'weight'] = weights.values
    
    return df

# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(df, portfolio_name, min_volume=5000):
    """Calculate portfolio metrics."""
    active = df[df['weight'] > 0].copy()
    
    if len(active) == 0:
        return None
    
    # Dead markets
    dead = active[active['volume'] < min_volume]
    
    metrics = {
        'Portfolio': portfolio_name,
        'Markets Included': len(active),
        'Markets Filtered': len(df) - len(active),
        'Filter Rate (%)': round(100 * (len(df) - len(active)) / len(df), 1),
        'Mean SRS': round(active['predicted_srs'].mean(), 4),
        'Median SRS': round(active['predicted_srs'].median(), 4),
        'Portfolio-Weighted SRS': round((active['weight'] * active['predicted_srs']).sum(), 4),
        'Mean Volume': active['volume'].mean(),
        'Median Volume': active['volume'].median(),
        'Dead Markets': len(dead),
        'Dead Market Rate (%)': round(100 * len(dead) / len(active), 1) if len(active) > 0 else 0,
    }
    
    return metrics

# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_analysis(df, thresholds=None):
    """Test different threshold values."""
    if thresholds is None:
        thresholds = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    
    results = []
    
    for tau in thresholds:
        port = clarity_portfolio(df, threshold=tau)
        active = port[port['weight'] > 0]
        
        if len(active) == 0:
            continue
        
        results.append({
            'Threshold (τ)': tau,
            'Markets': len(active),
            'Filter Rate (%)': round(100 * (len(df) - len(active)) / len(df), 1),
            'Mean SRS': round(active['predicted_srs'].mean(), 3),
            'Median Volume ($)': round(active['volume'].median(), 0),
        })
    
    return pd.DataFrame(results)

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(df, naive_df, clarity_df, threshold):
    """Create comparison visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. SRS Distribution
    ax1 = axes[0, 0]
    naive_active = naive_df[naive_df['weight'] > 0]['predicted_srs']
    clarity_active = clarity_df[clarity_df['weight'] > 0]['predicted_srs']
    
    ax1.hist(naive_active, bins=15, alpha=0.5, label='Naive', color='steelblue', density=True)
    ax1.hist(clarity_active, bins=15, alpha=0.5, label='Clarity', color='forestgreen', density=True)
    ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'τ = {threshold}')
    ax1.set_xlabel('Semantic Risk Score (SRS)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('SRS Distribution by Portfolio', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 2. Volume vs SRS (The Liquidity Cliff)
    ax2 = axes[0, 1]
    ax2.scatter(df['predicted_srs'], df['volume'] / 1000, alpha=0.5, color='steelblue', s=30)
    ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'τ = {threshold}')
    ax2.set_xlabel('Semantic Risk Score (SRS)', fontsize=11)
    ax2.set_ylabel('Volume ($K)', fontsize=11)
    ax2.set_title('The Liquidity Cliff: Volume vs SRS', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.set_yscale('log')
    
    # 3. Sensitivity Analysis
    ax3 = axes[1, 0]
    sens = sensitivity_analysis(df)
    ax3.plot(sens['Threshold (τ)'], sens['Median Volume ($)'] / 1000, 
             marker='o', linewidth=2, markersize=8, color='forestgreen')
    ax3.axvline(x=0.10, color='red', linestyle='--', alpha=0.7, label='Recommended τ')
    ax3.set_xlabel('Threshold (τ)', fontsize=11)
    ax3.set_ylabel('Median Volume ($K)', fontsize=11)
    ax3.set_title('Sensitivity: Median Volume by Threshold', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Markets Included vs Threshold
    ax4 = axes[1, 1]
    ax4.plot(sens['Threshold (τ)'], sens['Markets'], 
             marker='s', linewidth=2, markersize=8, color='steelblue')
    ax4.axvline(x=0.10, color='red', linestyle='--', alpha=0.7, label='Recommended τ')
    ax4.set_xlabel('Threshold (τ)', fontsize=11)
    ax4.set_ylabel('Markets Included', fontsize=11)
    ax4.set_title('Trade-off: Coverage vs Clarity', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('portfolio_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Plot saved as 'portfolio_comparison.png'")

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_backtest(input_path, threshold=0.15, gamma=1):
    """Run the complete backtest analysis."""
    
    # Load data
    df = load_data(input_path)
    
    # Construct portfolios
    print("=" * 70)
    print("CONSTRUCTING PORTFOLIOS")
    print("=" * 70)
    
    naive_df = naive_portfolio(df)
    clarity_df = clarity_portfolio(df, threshold=threshold, gamma=gamma)
    
    naive_active = naive_df[naive_df['weight'] > 0]
    clarity_active = clarity_df[clarity_df['weight'] > 0]
    
    print(f"Naive Portfolio:   {len(naive_active)} markets (all)")
    print(f"Clarity Portfolio: {len(clarity_active)} markets (SRS ≤ {threshold})")
    print()
    
    # Calculate metrics
    naive_metrics = calculate_metrics(naive_df, "Naive")
    clarity_metrics = calculate_metrics(clarity_df, f"Clarity (τ={threshold})")
    
    # Print comparison
    print("=" * 70)
    print("PORTFOLIO COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Metric':<30} {'Naive':>15} {'Clarity':>15} {'Change':>12}")
    print("-" * 72)
    
    # Markets
    n_naive = naive_metrics['Markets Included']
    n_clarity = clarity_metrics['Markets Included']
    pct_change = 100 * (n_clarity / n_naive - 1)
    print(f"{'Markets Included':<30} {n_naive:>15} {n_clarity:>15} {pct_change:>+11.0f}%")
    
    # Mean SRS
    srs_naive = naive_metrics['Mean SRS']
    srs_clarity = clarity_metrics['Mean SRS']
    pct_change = 100 * (srs_clarity / srs_naive - 1)
    print(f"{'Mean SRS':<30} {srs_naive:>15.3f} {srs_clarity:>15.3f} {pct_change:>+11.0f}%")
    
    # Portfolio-Weighted SRS
    wsrs_naive = naive_metrics['Portfolio-Weighted SRS']
    wsrs_clarity = clarity_metrics['Portfolio-Weighted SRS']
    pct_change = 100 * (wsrs_clarity / wsrs_naive - 1)
    print(f"{'Portfolio-Weighted SRS':<30} {wsrs_naive:>15.3f} {wsrs_clarity:>15.3f} {pct_change:>+11.0f}%")
    
    # Median Volume
    vol_naive = naive_metrics['Median Volume']
    vol_clarity = clarity_metrics['Median Volume']
    pct_change = 100 * (vol_clarity / vol_naive - 1)
    print(f"{'Median Volume':<30} {'$'+f'{vol_naive:,.0f}':>15} {'$'+f'{vol_clarity:,.0f}':>15} {pct_change:>+11.0f}%")
    
    # Mean Volume
    mvol_naive = naive_metrics['Mean Volume']
    mvol_clarity = clarity_metrics['Mean Volume']
    pct_change = 100 * (mvol_clarity / mvol_naive - 1)
    print(f"{'Mean Volume':<30} {'$'+f'{mvol_naive:,.0f}':>15} {'$'+f'{mvol_clarity:,.0f}':>15} {pct_change:>+11.0f}%")
    
    print()
    
    # Sensitivity Analysis
    print("=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)
    print()
    sens_df = sensitivity_analysis(df)
    print(sens_df.to_string(index=False))
    print()
    
    # Find optimal threshold (max median volume while keeping >50 markets)
    viable = sens_df[sens_df['Markets'] >= 50]
    if len(viable) > 0:
        best_row = viable.loc[viable['Median Volume ($)'].idxmax()]
        print(f"Recommended threshold: τ = {best_row['Threshold (τ)']} " +
              f"(Median Volume: ${best_row['Median Volume ($)']:,.0f}, " +
              f"Markets: {int(best_row['Markets'])})")
    print()
    
    # Create plots
    print("=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    create_plots(df, naive_df, clarity_df, threshold)
    
    # Summary for paper
    print()
    print("=" * 70)
    print("SUMMARY TABLE FOR PAPER (Copy this)")
    print("=" * 70)
    print()
    print("| Metric | Naive Portfolio | Clarity Portfolio | Improvement |")
    print("|--------|-----------------|-------------------|-------------|")
    print(f"| Markets Included | {n_naive} | {n_clarity} | {100*(n_clarity/n_naive - 1):+.0f}% |")
    print(f"| Mean SRS | {srs_naive:.3f} | {srs_clarity:.3f} | {100*(srs_clarity/srs_naive - 1):+.0f}% |")
    print(f"| Median Volume | ${vol_naive:,.0f} | ${vol_clarity:,.0f} | {100*(vol_clarity/vol_naive - 1):+.0f}% |")
    print(f"| Portfolio-Weighted SRS | {wsrs_naive:.3f} | {wsrs_clarity:.3f} | {100*(wsrs_clarity/wsrs_naive - 1):+.0f}% |")
    print()
    
    return df, naive_df, clarity_df, sens_df

# =============================================================================
# RUN THE ANALYSIS
# =============================================================================

if __name__ == "__main__":
    # Run with default threshold of 0.10 (empirically derived)
    df, naive_df, clarity_df, sens_df = run_backtest(
        input_path=INPUT_PATH,
        threshold=0.10,  # Use the empirically-derived threshold
        gamma=1
    )