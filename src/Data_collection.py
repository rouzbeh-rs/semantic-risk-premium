import requests
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import os
import math

# ============================================================================
# ⚠️ PASTE YOUR API KEY HERE ⚠️
# ============================================================================

GRAPH_API_KEY = ""  # <-- PASTE YOUR API KEY HERE

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "OUTPUT_DIR": "./prediction_market_data",

    # Collection limits - expanded to 100K
    "MAX_UMA_PRICE_REQUESTS": 100000,

    # Few-shot example settings
    # Distribution: ~87% low variance in real data
    # So we use 9 low-variance + 1 high-variance = 10 examples
    "NUM_FEWSHOT_LOW": 9,      # variance 0.00-0.05
    "NUM_FEWSHOT_HIGH": 1,     # variance > 0.10
    "RANDOM_SEED": 42,

    "REQUEST_DELAY": 0.5,
    "CHECKPOINT_EVERY": 1000,
    "BATCH_SIZE": 100,
}

# ============================================================================
# API ENDPOINTS
# ============================================================================

UMA_MAINNET_VOTING_V2_SUBGRAPH_ID = "5YVXjj28Lv4eLhHg54R1QWVHNn8VAjZnT3vJgEtuyWmY"

def get_uma_mainnet_url():
    if not GRAPH_API_KEY:
        return None
    return f"https://gateway.thegraph.com/api/{GRAPH_API_KEY}/subgraphs/id/{UMA_MAINNET_VOTING_V2_SUBGRAPH_ID}"

# ============================================================================
# SETUP & UTILITIES
# ============================================================================

def setup_environment():
    """Setup output directory."""
    output_dir = CONFIG["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir

def save_checkpoint(data: List[Dict], filename: str):
    """Save intermediate results."""
    filepath = os.path.join(CONFIG["OUTPUT_DIR"], filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, default=str)
    print(f"  Checkpoint: {filename} ({len(data)} records)")

def save_dataframe(df: pd.DataFrame, filename: str):
    """Save DataFrame to CSV."""
    filepath = os.path.join(CONFIG["OUTPUT_DIR"], filename)
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filename} ({len(df)} rows)")

def load_checkpoint(filename: str) -> Optional[List[Dict]]:
    """Load checkpoint if exists."""
    filepath = os.path.join(CONFIG["OUTPUT_DIR"], filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"  Resuming: {filename} ({len(data)} records)")
        return data
    return None

def query_subgraph(url: str, query: str) -> Dict:
    """Execute GraphQL query."""
    if not url:
        return {"error": "No URL (missing API key?)"}

    try:
        response = requests.post(url, json={"query": query}, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# 1. UMA DATA COLLECTION
# ============================================================================

def test_uma_connection() -> bool:
    """Test UMA Mainnet Voting V2 subgraph connection."""
    print("\nTesting UMA Mainnet Voting V2 connection...")

    if not GRAPH_API_KEY:
        print("  ERROR: No API key!")
        return False

    url = get_uma_mainnet_url()
    print(f"  URL: {url[:70]}...")

    result = query_subgraph(url, "{ priceRequests(first: 1) { id } }")

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return False
    if "errors" in result:
        print(f"  GraphQL ERROR: {result['errors']}")
        return False
    if result.get("data", {}).get("priceRequests") is not None:
        print("  SUCCESS!")
        return True

    print(f"  Unexpected response: {result}")
    return False

def fetch_uma_price_requests(first: int, skip: int) -> List[Dict]:
    """Fetch price requests from UMA Mainnet Voting V2."""
    query = f"""
    {{
      priceRequests(
        first: {first}
        skip: {skip}
        orderBy: time
        orderDirection: desc
      ) {{
        id
        isResolved
        price
        time
        resolutionTransaction
        resolutionTimestamp
        resolutionBlock
        resolvedPriceRequestIndex
        identifier {{
          id
          isSupported
        }}
        ancillaryData
        latestRound {{
          id
          roundId
          totalVotesRevealed
          groups {{
            price
            totalVoteAmount
          }}
        }}
        rounds {{
          id
          roundId
          totalVotesRevealed
        }}
      }}
    }}
    """
    result = query_subgraph(get_uma_mainnet_url(), query)

    if "error" in result:
        print(f"  Error: {result['error']}")
        return []
    if "errors" in result:
        print(f"  GraphQL Error: {result['errors']}")
        return []

    return result.get("data", {}).get("priceRequests", [])

def collect_uma_price_requests() -> List[Dict]:
    """Collect UMA Mainnet Voting price requests (disputes)."""
    max_requests = CONFIG["MAX_UMA_PRICE_REQUESTS"]
    print("\n" + "=" * 60)
    print(f"1. COLLECTING UMA DISPUTES (max: {max_requests:,})")
    print("=" * 60)

    if not test_uma_connection():
        return []

    all_requests = load_checkpoint("uma_checkpoint.json") or []

    if len(all_requests) >= max_requests:
        print(f"Already have {len(all_requests):,} requests.")
        return all_requests[:max_requests]

    skip = len(all_requests)
    batch_size = CONFIG["BATCH_SIZE"]
    empty_count = 0

    while len(all_requests) < max_requests:
        try:
            print(f"Fetching {skip:,} to {skip + batch_size:,}... ", end="")
            batch = fetch_uma_price_requests(batch_size, skip)

            if not batch:
                empty_count += 1
                print(f"Empty ({empty_count}/3)")
                if empty_count >= 3:
                    print("No more requests available.")
                    break
                time.sleep(2)
                continue

            empty_count = 0
            all_requests.extend(batch)
            print(f"Got {len(batch)}. Total: {len(all_requests):,}")

            skip += len(batch)
            time.sleep(CONFIG["REQUEST_DELAY"])

            if len(all_requests) % CONFIG["CHECKPOINT_EVERY"] < batch_size:
                save_checkpoint(all_requests, "uma_checkpoint.json")

        except Exception as e:
            print(f"Error: {e}")
            save_checkpoint(all_requests, "uma_checkpoint.json")
            time.sleep(5)

    save_checkpoint(all_requests, "uma_final.json")
    print(f"Complete: {len(all_requests):,} disputes collected")
    return all_requests

# ============================================================================
# 2. DATA PROCESSING
# ============================================================================

def decode_hex_to_string(hex_data: str) -> str:
    """Decode hex-encoded string to readable text."""
    if not hex_data:
        return ""
    try:
        if hex_data.startswith('0x'):
            hex_data = hex_data[2:]
        return bytes.fromhex(hex_data).decode('utf-8', errors='ignore')
    except:
        return ""

def decode_question_from_request_id(request_id: str) -> str:
    """Extract and decode question text from UMA request_id."""
    try:
        req_str = str(request_id)
        if '0x' in req_str:
            hex_part = req_str.split('0x', 1)[1]
            decoded = bytes.fromhex(hex_part).decode('utf-8', errors='ignore')
            return decoded
    except:
        pass
    return ""

def calculate_vote_entropy(vote_groups: List[Dict]) -> float:
    """Calculate Shannon entropy of vote distribution."""
    if not vote_groups:
        return 0.0

    amounts = []
    for g in vote_groups:
        try:
            amt = float(g.get('totalVoteAmount', 0))
            if amt > 0:
                amounts.append(amt)
        except:
            continue

    if not amounts:
        return 0.0

    total = sum(amounts)
    return -sum((a/total) * math.log2(a/total) for a in amounts)

def calculate_bernoulli_variance(vote_groups: List[Dict]) -> float:
    """Calculate vote split variance (0.25 = max contention at 50/50)."""
    if not vote_groups or len(vote_groups) < 2:
        return 0.0

    try:
        amounts = [float(g.get('totalVoteAmount', 0)) for g in vote_groups]
        total = sum(amounts)
        if total == 0:
            return 0.0
        p = amounts[0] / total
        return p * (1 - p)
    except:
        return 0.0

def process_uma_disputes(requests: List[Dict]) -> pd.DataFrame:
    """Process UMA disputes to DataFrame with decoded questions.
    
    NOTE: No deduplication is performed. Each resolution event is treated
    as an independent observation, even if question text is identical.
    """
    if not requests:
        return pd.DataFrame()

    print("\n" + "=" * 60)
    print("2. PROCESSING UMA DATA")
    print("=" * 60)

    rows = []
    for r in requests:
        latest = r.get('latestRound') or {}
        groups = latest.get('groups') or []
        ident = r.get('identifier') or {}
        request_id = r.get('id', '')
        ancillary_raw = r.get('ancillaryData', '')
        ancillary_decoded = decode_hex_to_string(ancillary_raw)

        # Decode question from request_id
        question_decoded = decode_question_from_request_id(request_id)

        rows.append({
            'uma_request_id': request_id,
            'uma_identifier': ident.get('id', ''),
            'uma_ancillary_data_decoded': ancillary_decoded,
            'question_decoded': question_decoded,
            'uma_timestamp': r.get('time', ''),
            'uma_resolution_timestamp': r.get('resolutionTimestamp', ''),
            'uma_is_resolved': r.get('isResolved', False),
            'uma_price': r.get('price', ''),
            'uma_total_votes_revealed': latest.get('totalVotesRevealed', 0),
            'uma_num_vote_groups': len(groups),
            'uma_num_rounds': len(r.get('rounds') or []),
            'uma_vote_entropy': calculate_vote_entropy(groups),
            'uma_bernoulli_variance': calculate_bernoulli_variance(groups),
            'uma_vote_groups_raw': json.dumps(groups),
        })

    df = pd.DataFrame(rows)

    # Filter to rows with actual question text (contains 'title:')
    has_question = df['question_decoded'].str.contains('title:', case=False, na=False)
    df_with_questions = df[has_question].copy()

    # NO DEDUPLICATION - each resolution event is independent
    
    print(f"  Total disputes: {len(df):,}")
    print(f"  With decoded questions: {len(df_with_questions):,}")
    print(f"  Unique question texts: {df_with_questions['question_decoded'].nunique():,}")

    return df_with_questions

# ============================================================================
# 3. EXTRACT FEW-SHOT EXAMPLES (MATCHING REAL DISTRIBUTION)
# ============================================================================

def extract_fewshot_examples(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract few-shot examples matching real data distribution.
    
    Real distribution is ~87% low variance (0.00-0.05).
    We select 9 low-variance + 1 high-variance = 10 examples.
    
    For few-shot selection, we DO deduplicate to avoid showing the model
    the same question text multiple times in the examples.
    
    Returns:
        fewshot_df: DataFrame with 10 few-shot examples
        analysis_df: DataFrame with remaining markets (few-shot excluded)
    """
    print("\n" + "=" * 60)
    print("3. EXTRACTING FEW-SHOT EXAMPLES")
    print("=" * 60)

    np.random.seed(CONFIG["RANDOM_SEED"])

    # Print variance distribution
    total = len(df)
    zero_var = (df['uma_bernoulli_variance'] == 0).sum()
    low_var = ((df['uma_bernoulli_variance'] > 0) & (df['uma_bernoulli_variance'] <= 0.05)).sum()
    med_var = ((df['uma_bernoulli_variance'] > 0.05) & (df['uma_bernoulli_variance'] <= 0.15)).sum()
    high_var = (df['uma_bernoulli_variance'] > 0.15).sum()

    print(f"\nVariance distribution in full dataset (n={total:,}):")
    print(f"  Zero (0.00):       {zero_var:,} ({100*zero_var/total:.1f}%)")
    print(f"  Low (0.00-0.05):   {low_var:,} ({100*low_var/total:.1f}%)")
    print(f"  Medium (0.05-0.15): {med_var:,} ({100*med_var/total:.1f}%)")
    print(f"  High (>0.15):      {high_var:,} ({100*high_var/total:.1f}%)")

    # For few-shot selection, deduplicate by question text
    # (we don't want to show the model the same text twice in examples)
    df_unique = df.drop_duplicates(subset='question_decoded', keep='first')
    print(f"\n  Unique questions for few-shot selection: {len(df_unique):,}")

    # Select low-variance examples (0.00-0.05)
    low_var_df = df_unique[df_unique['uma_bernoulli_variance'] <= 0.05].copy()
    num_low = CONFIG["NUM_FEWSHOT_LOW"]
    
    if len(low_var_df) < num_low:
        print(f"  WARNING: Only {len(low_var_df)} low-variance markets available")
        num_low = len(low_var_df)
    
    fewshot_low = low_var_df.sample(n=num_low, random_state=CONFIG["RANDOM_SEED"])

    # Select high-variance examples (>0.10 for clear signal)
    high_var_df = df_unique[df_unique['uma_bernoulli_variance'] > 0.10].copy()
    num_high = CONFIG["NUM_FEWSHOT_HIGH"]
    
    if len(high_var_df) < num_high:
        print(f"  WARNING: Only {len(high_var_df)} high-variance markets available")
        num_high = len(high_var_df)
    
    fewshot_high = high_var_df.sample(n=num_high, random_state=CONFIG["RANDOM_SEED"])

    # Combine few-shot examples
    fewshot_df = pd.concat([fewshot_low, fewshot_high], ignore_index=True)
    fewshot_ids = set(fewshot_df['uma_request_id'].tolist())
    fewshot_questions = set(fewshot_df['question_decoded'].tolist())

    print(f"\nFew-shot examples selected: {len(fewshot_df)}")
    print(f"  Low-variance (0.00-0.05): {num_low}")
    print(f"  High-variance (>0.10): {num_high}")

    # Create analysis set:
    # Exclude by BOTH request_id AND question_text
    # (removes the selected examples AND any duplicates of those questions)
    analysis_df = df[
        (~df['uma_request_id'].isin(fewshot_ids)) & 
        (~df['question_decoded'].isin(fewshot_questions))
    ].copy()

    print(f"\nAnalysis set: {len(analysis_df):,} markets")
    print(f"  (Excluded {len(df) - len(analysis_df)} rows: few-shot examples + their duplicates)")

    return fewshot_df, analysis_df

def format_fewshot_for_prompt(fewshot_df: pd.DataFrame) -> str:
    """Format few-shot examples for use in the predictor prompt."""
    
    print("\n" + "=" * 60)
    print("4. FORMATTING FEW-SHOT EXAMPLES FOR PROMPT")
    print("=" * 60)

    examples = []
    
    for idx, row in fewshot_df.iterrows():
        variance = row['uma_bernoulli_variance']
        question = row['question_decoded']
        
        # Extract title if present
        if 'title:' in question.lower():
            start = question.lower().find('title:') + 6
            # Find end of title (usually at comma or next field)
            end = len(question)
            for delimiter in [', description:', ',description:', ', res_data:', ',res_data:']:
                pos = question.lower().find(delimiter, start)
                if pos > 0:
                    end = min(end, pos)
            title = question[start:end].strip()
        else:
            title = question[:100].strip()
        
        # Truncate if too long
        if len(title) > 80:
            title = title[:77] + "..."
        
        examples.append(f'Example {len(examples)+1} (Variance: {variance:.2f}): "{title}"')

    formatted = "\n\n".join(examples)
    
    print("\nFormatted examples:")
    print("-" * 40)
    print(formatted)
    print("-" * 40)
    
    return formatted

def save_fewshot_prompt_file(formatted_examples: str):
    """Save the formatted few-shot examples to a file for use in predictor."""
    filepath = os.path.join(CONFIG["OUTPUT_DIR"], "fewshot_examples.txt")
    with open(filepath, 'w') as f:
        f.write(formatted_examples)
    print(f"\n  Saved: fewshot_examples.txt")

# ============================================================================
# 5. SUMMARY STATISTICS
# ============================================================================

def print_summary_stats(df: pd.DataFrame, label: str = "Dataset"):
    """Print summary statistics for a dataset."""
    
    print(f"\n{label} Statistics:")
    print(f"  Total markets: {len(df):,}")
    
    if 'uma_bernoulli_variance' in df.columns:
        var = df['uma_bernoulli_variance']
        print(f"  Variance: min={var.min():.4f}, max={var.max():.4f}, mean={var.mean():.4f}, median={var.median():.4f}")
        
        # Distribution
        zero = (var == 0).sum()
        low = ((var > 0) & (var <= 0.05)).sum()
        med = ((var > 0.05) & (var <= 0.15)).sum()
        high = (var > 0.15).sum()
        
        print(f"  Distribution:")
        print(f"    Zero (0.00):       {zero:,} ({100*zero/len(df):.1f}%)")
        print(f"    Low (0.00-0.05):   {low:,} ({100*low/len(df):.1f}%)")
        print(f"    Medium (0.05-0.15): {med:,} ({100*med/len(df):.1f}%)")
        print(f"    High (>0.15):      {high:,} ({100*high/len(df):.1f}%)")
    
    if 'question_decoded' in df.columns:
        unique_q = df['question_decoded'].nunique()
        print(f"  Unique question texts: {unique_q:,}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete data collection and processing."""

    print("=" * 70)
    print("UMA DISPUTE DATA COLLECTOR v11")
    print("Paper: 'The Semantic Risk Premium'")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check API key
    if not GRAPH_API_KEY:
        print("\n" + "!" * 60)
        print("ERROR: No Graph API key set!")
        print("Please paste your API key in GRAPH_API_KEY at the top.")
        print("Get your free key at: https://thegraph.com/studio/")
        print("!" * 60)
        return None

    setup_environment()

    # Step 1: Collect UMA disputes (up to 100K)
    uma_raw = collect_uma_price_requests()
    if not uma_raw:
        print("ERROR: No UMA data collected")
        return None

    # Step 2: Process and decode (NO DEDUPLICATION)
    uma_df = process_uma_disputes(uma_raw)
    save_dataframe(uma_df, "01_uma_all_markets.csv")

    # Step 3: Extract few-shot examples and create analysis set
    fewshot_df, analysis_df = extract_fewshot_examples(uma_df)
    save_dataframe(fewshot_df, "02_fewshot_examples.csv")
    save_dataframe(analysis_df, "03_analysis_set.csv")

    # Step 4: Format few-shot examples for prompt
    formatted_examples = format_fewshot_for_prompt(fewshot_df)
    save_fewshot_prompt_file(formatted_examples)

    # Step 5: Print summary statistics
    print_summary_stats(uma_df, "Full Dataset")
    print_summary_stats(fewshot_df, "Few-Shot Examples")
    print_summary_stats(analysis_df, "Analysis Set")

    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)

    print(f"\nOutput Directory: {CONFIG['OUTPUT_DIR']}")
    print("\nFiles Created:")
    for f in sorted(os.listdir(CONFIG["OUTPUT_DIR"])):
        if f.endswith('.csv') or f.endswith('.txt'):
            path = os.path.join(CONFIG["OUTPUT_DIR"], f)
            size = os.path.getsize(path) / 1024
            print(f"  {f}: {size:.1f} KB")

    print("\n" + "=" * 70)
    print("READY FOR VARIANCE PREDICTION")
    print("=" * 70)

    return {
        'uma_df': uma_df,
        'fewshot_df': fewshot_df,
        'analysis_df': analysis_df,
        'formatted_examples': formatted_examples,
    }

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    results = main()