import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional
import time
import re
import os
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model settings
    "MODEL_NAME": "Qwen/Qwen3-8B",
    "QUANTIZATION": "4bit",
    "MAX_NEW_TOKENS": 300,
    "TEMPERATURE": 0.0,

    # Paths
    "INPUT_PATH": "./polymarket_data/05_analysis_ready.csv",
    "OUTPUT_DIR": "./polymarket_srs_predictions",

    # Processing
    "CHECKPOINT_EVERY": 50,
}

# ============================================================================
# 10 FEW-SHOT EXAMPLES (From clean_up_script.py)
# ============================================================================

FEW_SHOT_EXAMPLES_TEXT = """
## REAL EXAMPLES FROM UMA DISPUTE DATA
## Distribution: 5 Zero + 2 Low + 2 Medium + 1 High

Example 1 (Variance: 0.00): "MLB: New York Mets vs. Atlanta Braves 2023-04-29"
Analysis: Sports game outcome with official MLB results. Binary win/loss with no ambiguity.
PREDICTED VARIANCE: 0.00

Example 2 (Variance: 0.00): "Evansville vs. UIC"
Analysis: Sports matchup with clear official outcome. No interpretation needed.
PREDICTED VARIANCE: 0.00

Example 3 (Variance: 0.00): "Will Aston Villa vs. Paris Saint Germain end in a draw?"
Analysis: Soccer match outcome - draw is clearly defined. Official result determines resolution.
PREDICTED VARIANCE: 0.00

Example 4 (Variance: 0.00): "Hamas leadership out of Qatar before Trump in office?"
Analysis: Verifiable event with specific deadline (inauguration date). Binary yes/no.
PREDICTED VARIANCE: 0.00

Example 5 (Variance: 0.00): "Ducks vs. Red Wings"
Analysis: NHL game outcome. Official score determines winner with no ambiguity.
PREDICTED VARIANCE: 0.00

Example 6 (Variance: 0.02): "Will the highest temperature in London be 53°F or below on April 2?"
Analysis: Specific temperature threshold, specific location, specific date. Minor ambiguity about which weather source to use.
PREDICTED VARIANCE: 0.02

Example 7 (Variance: 0.03): "U.S. military action against Yemen in 2024?"
Analysis: Observable military event, but "military action" could have varying interpretations (strikes vs advisors vs full engagement).
PREDICTED VARIANCE: 0.03

Example 8 (Variance: 0.08): "Will Trump launch a coin before the election?"
Analysis: "Launch a coin" is ambiguous - official crypto token? Meme coin? Physical commemorative coin? "Before election" needs specific date.
PREDICTED VARIANCE: 0.08

Example 9 (Variance: 0.10): "Will Trump lower tariffs on China in April?"
Analysis: "Lower tariffs" is vague - any reduction? Specific categories? "April" of which year? Multiple interpretations possible.
PREDICTED VARIANCE: 0.10

Example 10 (Variance: 0.21): "Farcaster unique users less than 15k on Feb 5?"
Analysis: "Unique users" definition unclear - daily active? Total registered? Which timezone for Feb 5? No official source specified.
PREDICTED VARIANCE: 0.21
"""

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = f"""You predict dispute variance for prediction market contracts.

SCALE:
- 0.00 = Perfect clarity, all resolvers would agree (e.g., sports scores)
- 0.25 = Maximum ambiguity, resolvers split 50/50

KEY INSIGHT: Most prediction market contracts (~87%) have variance at or near 0.00 because they are well-specified.

FACTORS THAT KEEP VARIANCE LOW (0.00-0.03):
- Sports outcomes (official scores)
- Election results (official counts)
- Specific price thresholds with named exchanges
- Exact dates and times
- Named official data sources

FACTORS THAT INCREASE VARIANCE (0.05+):
- Vague qualifiers ("significant", "major", "meaningful")
- No resolution source specified
- Subjective terms ("launch", "action", "improve")
- Complex multi-part conditions
- Edge cases not addressed

{FEW_SHOT_EXAMPLES_TEXT}

TASK: Analyze the contract below and predict its dispute variance.
Output: Brief analysis (1-2 sentences), then "PREDICTED VARIANCE: X.XX" on the final line."""

# ============================================================================
# GPU MANAGEMENT
# ============================================================================

def clear_gpu_memory():
    """Clear GPU memory."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load the quantized LLM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading model: {CONFIG['MODEL_NAME']}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["MODEL_NAME"],
        trust_remote_code=True,
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["MODEL_NAME"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return model, tokenizer

# ============================================================================
# PREDICTION
# ============================================================================

def predict_srs(model, tokenizer, contract_text: str) -> Dict:
    """Predict Semantic Risk Score for a contract."""

    contract_truncated = contract_text[:2000]

    user_prompt = f"""Analyze this prediction market contract and predict its dispute variance:

CONTRACT: {contract_truncated}

Remember: Most contracts have variance near 0.00 (like sports outcomes). Only predict higher if there's genuine ambiguity in the resolution criteria.

Brief analysis, then final line: "PREDICTED VARIANCE: X.XX"
/no_think"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Conditional generation based on temperature
        if CONFIG["TEMPERATURE"] > 0:
            # Sampling mode (non-deterministic)
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG["MAX_NEW_TOKENS"],
                temperature=CONFIG["TEMPERATURE"],
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            # Greedy decoding mode (deterministic, effectively temp=0)
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG["MAX_NEW_TOKENS"],
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id,
            )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Extract variance
    variance = extract_variance(response)

    return {
        "response": response[:500],
        "predicted_srs": variance
    }

def extract_variance(response: str) -> Optional[float]:
    """Extract variance from model response."""

    patterns = [
        r'PREDICTED\s*VARIANCE:\s*(0\.\d+)',
        r'PREDICTED\s*VARIANCE:\s*(0)',
        r'[Vv]ariance:\s*(0\.\d+)',
        r'\*\*(0\.\d+)\*\*',
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                value = float(match.group(1))
                return max(0.0, min(0.25, value))
            except:
                continue

    # Fallback: find decimal in last part of response
    decimals = re.findall(r'(0\.\d+)', response[-100:])
    for d in reversed(decimals):
        try:
            value = float(d)
            if 0.0 <= value <= 0.30:
                return min(0.25, value)
        except:
            continue

    return None

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_srs_prediction():
    """Run SRS prediction on Polymarket contracts."""

    print("=" * 70)
    print("SRS PREDICTION FOR POLYMARKET")
    print("Using Your 10 UMA Few-Shot Examples")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Setup
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    # Load Polymarket data
    print(f"\nLoading Polymarket data from: {CONFIG['INPUT_PATH']}")
    if not os.path.exists(CONFIG['INPUT_PATH']):
        print(f"ERROR: Input file not found: {CONFIG['INPUT_PATH']}")
        print("Please run pricing_01_polymarket_collection.py first.")
        return None

    df = pd.read_csv(CONFIG["INPUT_PATH"])
    print(f"Loaded {len(df)} markets")

    # Show few-shot examples being used
    print("\n" + "=" * 70)
    print("FEW-SHOT EXAMPLES (from your UMA selection)")
    print("=" * 70)
    print("Zero variance: MLB Mets/Braves, Evansville/UIC, Aston Villa/PSG, Hamas/Qatar, Ducks/Red Wings")
    print("Low variance: London temperature, Yemen military action")
    print("Medium variance: Trump coin, Trump tariffs")
    print("High variance: Farcaster users")
    print("=" * 70)

    # Load model
    print("\n--- Loading Model ---")
    clear_gpu_memory()
    model, tokenizer = load_model()

    # Run predictions
    print("\n--- Running Predictions ---")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting SRS"):
        condition_id = row.get('condition_id', f'market_{idx}')
        question = row.get('question', '')
        description = row.get('description', '')

        # Combine question and description
        contract_text = f"Question: {question}\n\nDescription: {description}"

        if len(contract_text.strip()) < 20:
            results.append({
                'condition_id': condition_id,
                'predicted_srs': None,
                'response': 'SKIPPED: No contract text'
            })
            continue

        # Predict
        start_time = time.time()
        result = predict_srs(model, tokenizer, contract_text)
        elapsed = time.time() - start_time

        results.append({
            'condition_id': condition_id,
            'predicted_srs': result['predicted_srs'],
            'response': result['response'],
            'elapsed_seconds': elapsed
        })

        # Checkpoint
        if len(results) % CONFIG["CHECKPOINT_EVERY"] == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_path = os.path.join(CONFIG["OUTPUT_DIR"], f"checkpoint_{len(results)}.csv")
            checkpoint_df.to_csv(checkpoint_path, index=False)
            print(f"\n  Checkpoint saved: {len(results)} predictions")

    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(CONFIG["OUTPUT_DIR"], "srs_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved predictions: {results_path}")

    # Merge with original data
    final_df = df.merge(results_df[['condition_id', 'predicted_srs']], on='condition_id', how='left')
    final_path = os.path.join(CONFIG["OUTPUT_DIR"], "analysis_with_srs.csv")
    final_df.to_csv(final_path, index=False)
    print(f"Saved merged data: {final_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid_predictions = results_df['predicted_srs'].notna().sum()
    print(f"Valid predictions: {valid_predictions}/{len(results_df)}")

    if valid_predictions > 0:
        srs = results_df['predicted_srs'].dropna()
        print(f"\nSRS Statistics:")
        print(f"  Mean:   {srs.mean():.4f}")
        print(f"  Median: {srs.median():.4f}")
        print(f"  Std:    {srs.std():.4f}")
        print(f"  Min:    {srs.min():.4f}")
        print(f"  Max:    {srs.max():.4f}")

        print(f"\nSRS Distribution:")
        print(f"  Zero (0.00):       {(srs == 0).sum()} ({100*(srs == 0).sum()/len(srs):.1f}%)")
        print(f"  Low (0.00-0.05):   {((srs > 0) & (srs <= 0.05)).sum()} ({100*((srs > 0) & (srs <= 0.05)).sum()/len(srs):.1f}%)")
        print(f"  Medium (0.05-0.15): {((srs > 0.05) & (srs <= 0.15)).sum()} ({100*((srs > 0.05) & (srs <= 0.15)).sum()/len(srs):.1f}%)")
        print(f"  High (>0.15):      {(srs > 0.15).sum()} ({100*(srs > 0.15).sum()/len(srs):.1f}%)")

    print("\n" + "=" * 70)
    print("NEXT STEP: Run pricing_03_regressions.py")
    print("=" * 70)

    return final_df

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    df = run_srs_prediction()