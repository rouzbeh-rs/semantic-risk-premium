import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "OUTPUT_DIR": "./polymarket_data",
    "GAMMA_API_BASE": "https://gamma-api.polymarket.com",
    "CLOB_API_BASE": "https://clob.polymarket.com",

    # PILOT SETTINGS
    "MAX_MARKETS": 1000,
    "OFFSET_START": 8000,

    "REQUEST_DELAY": 0.2,
    "CHECKPOINT_EVERY": 100,
    "MIN_VOLUME": 5000,
    "MIN_DAYS_ACTIVE": 3,
    "MIN_YEAR": 2022,
}

# ============================================================================
# SETUP
# ============================================================================

def setup():
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    print(f"Output directory: {CONFIG['OUTPUT_DIR']}")

def fetch_markets(limit: int = 100, offset: int = 0, closed: bool = True) -> List[Dict]:
    url = f"{CONFIG['GAMMA_API_BASE']}/markets"
    params = {"limit": limit, "offset": offset, "closed": str(closed).lower()}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except:
        return []

def fetch_all_resolved_markets() -> List[Dict]:
    print("\n" + "=" * 70)
    print(f"FETCHING MARKETS (Offset: {CONFIG['OFFSET_START']}+, Max: {CONFIG['MAX_MARKETS']})")
    print("=" * 70)

    # Check for existing raw file
    raw_path = os.path.join(CONFIG["OUTPUT_DIR"], "01_raw_markets_pilot.json")
    if os.path.exists(raw_path):
        print(f"Loading existing raw data: {raw_path}")
        with open(raw_path, 'r') as f:
            return json.load(f)

    all_markets = []
    offset = CONFIG["OFFSET_START"]
    limit = 100
    max_offset = offset + 50000

    while len(all_markets) < CONFIG['MAX_MARKETS'] and offset < max_offset:
        batch = fetch_markets(limit=limit, offset=offset, closed=True)
        if not batch: break

        valid_batch = []
        for m in batch:
            try:
                created = m.get('createdAt') or m.get('created_at')
                if created and int(created[:4]) >= CONFIG['MIN_YEAR']:
                    valid_batch.append(m)
            except: pass

        all_markets.extend(valid_batch)
        print(f"  Offset {offset}: Fetched {len(batch)}. Kept {len(valid_batch)} (2022+). Total: {len(all_markets):,}")

        offset += limit
        time.sleep(CONFIG["REQUEST_DELAY"])

        if len(batch) < limit: break

    all_markets = all_markets[:CONFIG['MAX_MARKETS']]

    with open(raw_path, 'w') as f:
        json.dump(all_markets, f)
    return all_markets

# ============================================================================
# CHUNKED PRICE FETCHING
# ============================================================================

def fetch_price_history_chunked(token_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Fetches price history in 30-day chunks.
    """
    url = f"{CONFIG['CLOB_API_BASE']}/prices-history"
    CHUNK_SIZE = 30 * 24 * 60 * 60

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    all_history = []
    current_start = start_ts

    while current_start < end_ts:
        current_end = min(current_start + CHUNK_SIZE, end_ts)

        if current_end - current_start < 60:
            break

        params = {
            "market": token_id,
            "startTs": current_start,
            "endTs": current_end,
            "fidelity": 60
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                chunk_history = data.get("history", [])
                all_history.extend(chunk_history)
        except:
            pass

        current_start = current_end
        time.sleep(0.1)

    if not all_history:
        return pd.DataFrame()

    df = pd.DataFrame(all_history)
    if 't' in df.columns and 'p' in df.columns:
        df['timestamp'] = pd.to_datetime(df['t'], unit='s')
        df['price'] = df['p'].astype(float)
        df = df.sort_values('timestamp').drop_duplicates('timestamp')
        return df

    return pd.DataFrame()

# ============================================================================
# LOGIC
# ============================================================================

def process_market(market: Dict) -> Optional[Dict]:
    try:
        question = market.get('question', '')

        yes_token_id = None
        # FIX: Get outcome from root level first
        outcome_resolved = market.get('outcome')

        tokens = market.get('tokens', [])
        if isinstance(tokens, str):
            try: tokens = json.loads(tokens)
            except: tokens = []

        if tokens and isinstance(tokens, list):
            for token in tokens:
                if not isinstance(token, dict): continue
                outcome_label = str(token.get('outcome', '')).lower()
                tid = token.get('token_id') or token.get('tokenId')

                # Fallback: check winner status in token
                if not outcome_resolved and token.get('winner') is True:
                    outcome_resolved = token.get('outcome')

                if outcome_label == 'yes':
                    yes_token_id = tid

        if not yes_token_id:
            # Fallback ID extraction
            clob_ids = market.get('clobTokenIds', [])
            outcomes = market.get('outcomes', [])
            if isinstance(clob_ids, str):
                try: clob_ids = json.loads(clob_ids)
                except: clob_ids = []
            if isinstance(outcomes, str):
                try: outcomes = json.loads(outcomes)
                except: outcomes = []

            if clob_ids and outcomes and len(clob_ids) == len(outcomes):
                for idx, out_val in enumerate(outcomes):
                    if str(out_val).lower() == 'yes':
                        yes_token_id = clob_ids[idx]

        if not yes_token_id: return None

        return {
            'condition_id': market.get('conditionId'),
            'question': question,
            'yes_token_id': yes_token_id,
            'volume': float(market.get('volume', 0)),
            'created_at': market.get('createdAt'),
            'resolved_at': market.get('endDate'),
            'outcome_resolved': outcome_resolved
        }
    except: return None

def calculate_metrics(price_df: pd.DataFrame, outcome_res: str) -> Dict:
    metrics = {
        'price_count': len(price_df),
        'final_price': None,
        'volatility_24h': None,
        'resolution_jump': None
    }

    if price_df.empty: return metrics

    metrics['final_price'] = price_df['price'].iloc[-1]

    try:
        price_df['returns'] = price_df['price'].pct_change()

        # FIX: Use LAST 24 DATA POINTS regardless of timestamp
        # (Assuming fidelity=60 means 1 point per hour roughly, 24 points = 24 hours)
        if len(price_df) > 5:
            # Take the tail (last 24 points or less if not available)
            window_size = min(len(price_df), 24)
            tail_df = price_df.tail(window_size)
            metrics['volatility_24h'] = tail_df['returns'].std()

        # Resolution Jump
        # Normalize outcome string
        out_res = str(outcome_res).lower() if outcome_res else ""

        target = None
        if out_res == 'yes': target = 1.0
        elif out_res == 'no': target = 0.0

        if target is not None:
            metrics['resolution_jump'] = abs(target - metrics['final_price'])

    except: pass

    return metrics

# ============================================================================
# MAIN
# ============================================================================

def collect_all_data():
    print("=" * 70)
    print("POLYMARKET DATA COLLECTION (v10 - CALCULATION FIX)")
    print("=" * 70)
    setup()

    raw_markets = fetch_all_resolved_markets()

    print("\nPROCESSING...")
    processed = []
    for m in tqdm(raw_markets):
        p = process_market(m)
        if p: processed.append(p)

    df = pd.DataFrame(processed)
    print(f"Valid markets found: {len(df)}")

    if df.empty: return

    df = df[df['volume'] >= CONFIG['MIN_VOLUME']]
    print(f"After volume filter (>${CONFIG['MIN_VOLUME']}): {len(df)}")

    if df.empty: return

    print("\nFETCHING PRICES (CHUNKED)...")
    metrics_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            end_ts = int(pd.to_datetime(row['resolved_at']).timestamp())
            start_ts = int(pd.to_datetime(row['created_at']).timestamp())
        except:
            metrics_list.append({})
            continue

        price_df = fetch_price_history_chunked(row['yes_token_id'], start_ts, end_ts)
        m = calculate_metrics(price_df, row['outcome_resolved'])
        m['condition_id'] = row['condition_id']
        metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list)
    final_df = df.merge(metrics_df, on='condition_id', how='left')

    valid_vol = final_df['volatility_24h'].notna().sum()
    valid_jump = final_df['resolution_jump'].notna().sum()

    print("\n" + "=" * 70)
    print("DATA QUALITY REPORT")
    print(f"Total Markets:       {len(final_df)}")
    print(f"With Volatility:     {valid_vol} ({100*valid_vol/len(final_df):.1f}%)")
    print(f"With Resolution Jump:{valid_jump} ({100*valid_jump/len(final_df):.1f}%)")
    print("=" * 70)

    out_path = os.path.join(CONFIG["OUTPUT_DIR"], "05_analysis_ready.csv")
    final_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    collect_all_data()