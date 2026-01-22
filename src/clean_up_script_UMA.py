import pandas as pd
import re
import os

# 1. Configuration
# ----------------------------------------------------------------------------
DATA_DIR = "prediction_market_data"
INPUT_FILE = os.path.join(DATA_DIR, "01_uma_all_markets.csv")
TRAIN_OUTPUT = os.path.join(DATA_DIR, "02_fewshot_examples_v11.csv")
TEST_OUTPUT = os.path.join(DATA_DIR, "03_analysis_set.csv")

# 2. Load the full UMA dataset
# ----------------------------------------------------------------------------
if not os.path.exists(INPUT_FILE):
    print(f"❌ ERROR: Could not find '{INPUT_FILE}'")
    print(f"   Please ensure you have created the folder '{DATA_DIR}' and uploaded the CSV there.")
    # Fallback check: check current directory just in case
    if os.path.exists("01_uma_all_markets.csv"):
        print("   (Found it in the root directory instead. Using that.)")
        INPUT_FILE = "01_uma_all_markets.csv"
        df = pd.read_csv(INPUT_FILE)
    else:
        exit()
else:
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df)} markets from {INPUT_FILE}")

# Helper to normalize titles for matching (removes punctuation/casing issues)
def normalize_title(text):
    if not isinstance(text, str): return ""
    # Extract title part if "title:" exists
    match = re.search(r"title:\s*(.*?)(?:,|description:|$)", text, re.IGNORECASE | re.DOTALL)
    raw_title = match.group(1) if match else text
    return re.sub(r'[^a-zA-Z0-9]', '', raw_title).lower()

df['normalized_title'] = df['question_decoded'].apply(normalize_title)

# 3. Define the 10 Few-Shot Examples (5/2/2/1 Distribution)
# ----------------------------------------------------------------------------
# Includes the London Temperature replacement for the "Fart" market
few_shot_titles = [
    "MLB: New York Mets vs. Atlanta Braves 2023-04-29",           # Zero
    "Evansville vs. UIC",                                         # Zero
    "Will Aston Villa vs. Paris Saint Germain end in a draw?",    # Zero
    "Hamas leadership out of Qatar before Trump in office?",      # Zero
    "Ducks vs. Red Wings",                                        # Zero
    "Will the highest temperature in London be 53°F or below on April 2?", # Low (Replacement)
    "U.S. military action against Yemen in 2024?",                # Low
    "Will Trump launch a coin before the election?",              # Medium
    "Will Trump lower tariffs on China in April?",                # Medium
    "Farcaster unique users less than 15k on Feb 5?"              # High
]

# Create a normalized set for fast matching
target_hashes = set(re.sub(r'[^a-zA-Z0-9]', '', t).lower() for t in few_shot_titles)

# 4. Perform the Split
# ----------------------------------------------------------------------------
is_few_shot = df['normalized_title'].isin(target_hashes)

few_shot_df = df[is_few_shot].copy()
analysis_df = df[~is_few_shot].copy()

# 5. Verification & Saving
# ----------------------------------------------------------------------------
print(f"\n--- Split Summary ---")
print(f"Few-Shot Set (Training): {len(few_shot_df)} markets")
print(f"Analysis Set (Testing):  {len(analysis_df)} markets")
print(f"Total:                   {len(df)} markets")

# Verify we found all 10
if len(few_shot_df) < 10:
    print(f"\n⚠️ WARNING: Only found {len(few_shot_df)} out of 10 examples.")
    print("Missing titles:")
    found = set(few_shot_df['normalized_title'])
    for t in few_shot_titles:
        norm_t = re.sub(r'[^a-zA-Z0-9]', '', t).lower()
        if norm_t not in found:
            print(f" - {t}")
else:
    print("\n✅ All 10 few-shot examples successfully identified and isolated.")

# Save files
few_shot_df.drop(columns=['normalized_title'], inplace=True, errors='ignore')
analysis_df.drop(columns=['normalized_title'], inplace=True, errors='ignore')

few_shot_df.to_csv(TRAIN_OUTPUT, index=False)
analysis_df.to_csv(TEST_OUTPUT, index=False)

print(f"\nFiles Saved to {DATA_DIR}/:")
print(f" -> {os.path.basename(TRAIN_OUTPUT)} (Reference)")
print(f" -> {os.path.basename(TEST_OUTPUT)} (Use this for prediction)")