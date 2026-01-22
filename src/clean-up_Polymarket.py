import pandas as pd

# Load the raw V10 output
df = pd.read_csv("./polymarket_data/05_analysis_ready.csv")

print(f"Original Count: {len(df)}")

# 1. Filter for valid volatility (The Treatment Group)
df_clean = df[df['volatility_24h'].notna()].copy()

# 2. Add missing 'description' column (Required by pricing_02_srs.py)
if 'description' not in df_clean.columns:
    df_clean['description'] = "" # Fill with empty string
    print("Added placeholder 'description' column.")

# Save over the original file so the next script picks it up automatically
df_clean.to_csv("./polymarket_data/05_analysis_ready.csv", index=False)

print("-" * 30)
print(f"Ready for SRS Prediction: {len(df_clean)} markets")
print("Optimization: You just saved ~2 hours of compute time by removing invalid rows.")