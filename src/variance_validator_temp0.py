import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
import time
import re
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model settings - Qwen3-8B
    "MODEL_NAME": "Qwen/Qwen3-8B",
    "QUANTIZATION": "4bit",
    "MAX_NEW_TOKENS": 300,
    "TEMPERATURE": 0.0,

    # Prediction settings
    "NUM_PREDICTIONS": 1,

    # Output
    "OUTPUT_DIR": "./variance_predictions",
    "SAVE_INTERMEDIATE": True,

    # Default paths
    "DEFAULT_TEST_PATH": "prediction_market_data/01_uma_all_markets.csv",
}

# ============================================================================
# FEW-SHOT EXAMPLES - REAL EXAMPLES FROM UMA DATA (STRATIFIED SAMPLE)
# ============================================================================

FEW_SHOT_EXAMPLES = """
Example 1 (Variance: 0.00): "MLB: New York Mets vs. Atlanta Braves 2023-04-29"
- Binary sports outcome. Zero variance as official MLB scores are authoritative.

Example 2 (Variance: 0.00): "Evansville vs. UIC"
- NCAA Basketball match. Zero variance. Official sports results are binary and verifiable.

Example 3 (Variance: 0.00): "Will Aston Villa vs. Paris Saint Germain end in a draw?"
- Specific soccer outcome. Zero variance. Official match results are authoritative.

Example 4 (Variance: 0.00): "Hamas leadership out of Qatar before Trump in office?"
- Geopolitical event with clear physical criteria. Zero variance implies consensus on location status.

Example 5 (Variance: 0.00): "Ducks vs. Red Wings"
- NHL Hockey match. Zero variance. Official league results are authoritative.

Example 6 (Variance: 0.02): "Will the highest temperature in London be 53°F or below on April 2?"
- Low Variance (0.02). Specific numeric threshold and date. Minor variance typically stems from debates over which specific weather station is authoritative if not explicitly named.

Example 7 (Variance: 0.03): "U.S. military action against Yemen in 2024?"
- Low Variance (0.03). "Military action" is a broad term, but major events are usually distinct enough to form consensus.

Example 8 (Variance: 0.10): "Will Trump launch a coin before the election?"
- Medium Variance (0.10). Ambiguity likely centered on the definition of "launch" (announcement vs trading) or official vs unofficial affiliation.

Example 9 (Variance: 0.08): "Will Trump lower tariffs on China in April?"
- Medium Variance (0.08). Economic policy questions often face ambiguity regarding the exact timing of "lowering" vs announcement of intent.

Example 10 (Variance: 0.22): "Farcaster unique users less than 15k on Feb 5?"
- High Variance (0.22). Metric disputes often arise when different data providers (e.g., Dune Analytics vs direct node) show conflicting numbers.
"""

# ============================================================================
# SYSTEM PROMPT (CONDENSED)
# ============================================================================

SYSTEM_PROMPT = f"""You predict dispute variance for prediction market contracts.

SCALE:
- 0.00 = All resolvers agree (perfectly clear)
- 0.25 = 50/50 split (maximum ambiguity)

INCREASES VARIANCE: Vague terms, no resolution source, complex conditions, edge cases unaddressed.
DECREASES VARIANCE: Specific thresholds, named sources with URLs, clear definitions.

REAL EXAMPLES:
{FEW_SHOT_EXAMPLES}

Output format: Brief analysis (2-3 sentences), then "PREDICTED VARIANCE: X.XX" on final line."""

# ============================================================================
# GPU MEMORY MANAGEMENT
# ============================================================================

def clear_gpu_memory():
    """Clear GPU memory before loading model."""
    import gc

    for name in list(globals().keys()):
        try:
            obj = globals()[name]
            if isinstance(obj, torch.nn.Module):
                del globals()[name]
        except:
            pass

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if torch.cuda.is_available():
        print(f"GPU memory after clearing: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load the quantized LLM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading model: {CONFIG['MODEL_NAME']}")
    print(f"Quantization: {CONFIG['QUANTIZATION']}")

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

    print(f"Model loaded on {model.device}")
    print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    return model, tokenizer

# ============================================================================
# PREDICTION
# ============================================================================

def predict_variance(
    model,
    tokenizer,
    contract_text: str,
    temperature: float = 0.3
) -> Dict:
    """Generate a single variance prediction for a contract."""

    # Truncate contract to avoid token limits
    contract_truncated = contract_text[:1500]

    # Use /no_think to disable Qwen3 thinking mode
    user_prompt = f"""Predict dispute variance for this contract:

CONTRACT: {contract_truncated}

Brief analysis, then final line must be: "PREDICTED VARIANCE: X.XX"
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

    # --- INDENTATION FIXED BELOW ---
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

    return {
        "response": response,
        "predicted_variance": extract_variance(response)
    }


def extract_variance(response: str) -> Optional[float]:
    """Extract predicted variance from response."""

    # Look for "PREDICTED VARIANCE: X.XX" pattern
    patterns = [
        r'PREDICTED\s*VARIANCE:\s*(0\.\d+)',
        r'PREDICTED\s*VARIANCE:\s*(0)',
        r'[Pp]redicted\s*[Vv]ariance:\s*(0\.\d+)',
        r'[Vv]ariance:\s*(0\.\d+)',
        r'\*\*(0\.\d+)\*\*',
        r':\s*(0\.\d{2})\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                value = float(match.group(1))
                return max(0.0, min(0.25, value))
            except:
                continue

    # Fallback: look for any decimal between 0 and 0.25 in last 100 chars
    last_part = response[-100:]
    decimals = re.findall(r'(0\.\d+)', last_part)
    for d in reversed(decimals):
        try:
            value = float(d)
            if 0.0 <= value <= 0.30:
                return min(0.25, value)
        except:
            continue

    print("    Warning: Could not extract variance")
    return None


def predict_with_averaging(
    model,
    tokenizer,
    contract_text: str,
    num_predictions: int = 3
) -> Dict:
    """Make multiple predictions and average them for stability."""

    predictions = []
    responses = []

    for i in range(num_predictions):
        # --- FIX STARTS HERE ---
        # Explicitly pass the temperature from your CONFIG
        result = predict_variance(
            model, 
            tokenizer, 
            contract_text, 
            temperature=CONFIG["TEMPERATURE"]
        )
        # --- FIX ENDS HERE ---

        responses.append(result["response"])

        if result["predicted_variance"] is not None:
            predictions.append(result["predicted_variance"])
            print(f"    [{i+1}/{num_predictions}] Predicted: {result['predicted_variance']:.4f}")
        else:
            print(f"    [{i+1}/{num_predictions}] Failed to extract")

    if not predictions:
        return {
            "predicted_variance": None,
            "variance_std": None,
            "num_valid": 0,
            "responses": responses,
        }

    return {
        "predicted_variance": np.mean(predictions),
        "variance_std": np.std(predictions),
        "num_valid": len(predictions),
        "all_predictions": predictions,
        "responses": responses,
    }

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_market(
    model,
    tokenizer,
    market_text: str,
    market_id: str = "",
) -> Dict:
    """Analyze a single market and predict variance."""

    print(f"\nAnalyzing market: {market_id[:50]}...")
    start_time = time.time()

    result = predict_with_averaging(
        model, tokenizer, market_text,
        num_predictions=CONFIG["NUM_PREDICTIONS"]
    )

    elapsed = time.time() - start_time

    output = {
        "market_id": market_id,
        "market_text": market_text[:500],
        "predicted_variance": result["predicted_variance"],
        "variance_std": result["variance_std"],
        "num_valid_predictions": result["num_valid"],
        "elapsed_seconds": elapsed,
    }

    if result["predicted_variance"] is not None:
        print(f"  Predicted Variance: {result['predicted_variance']:.4f} (std={result['variance_std']:.4f})")
    else:
        print(f"  Prediction FAILED")
    print(f"  Time: {elapsed:.1f}s")

    return output


def run_analysis(
    test_data_path: str = None,
    output_dir: str = "./variance_predictions_v9",
    max_markets: int = None
):
    """Run variance prediction on test dataset."""

    if test_data_path is None:
        test_data_path = CONFIG["DEFAULT_TEST_PATH"]

    print("=" * 70)
    print("SEMANTIC RISK PREDICTOR v9")
    print("Direct Variance Prediction with Few-Shot Learning")
    print("Paper: 'The Semantic Risk Premium'")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print(f"\nApproach:")
    print(f"  - Direct prediction of Bernoulli variance (0.00-0.25)")
    print(f"  - 10 REAL few-shot examples from UMA dispute data")
    print(f"  - {CONFIG['NUM_PREDICTIONS']} predictions averaged per market")
    print(f"  - Thinking mode disabled for faster inference")

    print(f"\nConfiguration:")
    print(f"  Model: {CONFIG['MODEL_NAME']}")
    print(f"  Quantization: {CONFIG['QUANTIZATION']}")
    print(f"  Temperature: {CONFIG['TEMPERATURE']}")

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("\n--- Loading Model ---")
    clear_gpu_memory()
    model, tokenizer = load_model()

    # Load test data
    print(f"\n--- Loading Test Data ---")
    print(f"Path: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    print(f"Loaded {len(test_df)} markets")

    if max_markets:
        test_df = test_df.head(max_markets)
        print(f"Limited to {max_markets} markets")

    est_minutes = len(test_df) * CONFIG["NUM_PREDICTIONS"] * 0.3
    print(f"Estimated runtime: ~{est_minutes:.0f} minutes")

    # Run analysis
    print(f"\n--- Running Analysis ---")
    results = []

    for idx, row in test_df.iterrows():
        market_id = row.get('uma_request_id', f'market_{idx}')
        market_text = row.get('question_decoded', '')

        if not market_text or len(market_text) < 20:
            print(f"Skipping {idx}: No question text")
            continue

        result = analyze_market(
            model, tokenizer,
            market_text,
            market_id=str(market_id)[:100],
        )

        # Add ground truth
        result['ground_truth_variance'] = row.get('uma_bernoulli_variance', None)
        result['variance_bin'] = row.get('variance_bin', None)

        results.append(result)

        # Save intermediate
        if CONFIG["SAVE_INTERMEDIATE"]:
            intermediate_path = os.path.join(output_dir, f"result_{idx}.json")
            with open(intermediate_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)

        # Progress
        completed = len(results)
        remaining = len(test_df) - completed
        if results:
            avg_time = sum(r['elapsed_seconds'] for r in results) / len(results)
            est_remaining = remaining * avg_time / 60
            print(f"  Progress: {completed}/{len(test_df)} | Est. remaining: {est_remaining:.1f} min")

    # Save results
    print(f"\n--- Saving Results ---")

    summary_data = []
    for r in results:
        gt = r['ground_truth_variance']
        pred = r['predicted_variance']
        error = abs(pred - gt) if pred is not None and gt is not None else None

        summary_data.append({
            'market_id': r['market_id'][:50],
            'predicted_variance': pred,
            'ground_truth_variance': gt,
            'variance_bin': r['variance_bin'],
            'prediction_std': r['variance_std'],
            'prediction_error': error,
            'elapsed_seconds': r['elapsed_seconds'],
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "prediction_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Validation
    print(f"\n--- Validation ---")
    valid_mask = summary_df['predicted_variance'].notna() & summary_df['ground_truth_variance'].notna()

    if valid_mask.sum() >= 3:
        from scipy.stats import spearmanr, pearsonr

        predicted = summary_df.loc[valid_mask, 'predicted_variance']
        actual = summary_df.loc[valid_mask, 'ground_truth_variance']

        spearman_corr, spearman_p = spearmanr(predicted, actual)
        pearson_corr, pearson_p = pearsonr(predicted, actual)
        mae = summary_df.loc[valid_mask, 'prediction_error'].mean()

        print(f"Sample size: {valid_mask.sum()}")
        print(f"\nCorrelation (Predicted vs Actual Variance):")
        print(f"  Spearman: {spearman_corr:.4f} (p={spearman_p:.4f})")
        print(f"  Pearson:  {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"\nMean Absolute Error: {mae:.4f}")

        print(f"\n--- Interpretation ---")
        if spearman_p < 0.05 and spearman_corr > 0:
            print("✓ SUCCESS: Significant positive correlation!")
            print("  Model predictions align with actual dispute rates.")
        elif spearman_p < 0.10 and spearman_corr > 0:
            print("~ MARGINAL: Positive trend, marginally significant.")
        elif spearman_corr > 0:
            print("? INCONCLUSIVE: Positive correlation but not significant.")
        else:
            print("✗ NO RELATIONSHIP: Predictions do not correlate with actual variance.")

    # Results by variance bin
    print(f"\n--- Results by Variance Bin ---")
    if 'variance_bin' in summary_df.columns and valid_mask.any():
        bin_summary = summary_df[valid_mask].groupby('variance_bin').agg({
            'predicted_variance': 'mean',
            'ground_truth_variance': 'mean',
            'prediction_error': 'mean',
        }).round(4)
        print(bin_summary.to_string())

    # Individual predictions
    print(f"\n--- Individual Predictions ---")
    for _, row in summary_df.iterrows():
        pred = row['predicted_variance']
        actual = row['ground_truth_variance']
        vbin = row['variance_bin']
        pred_str = f"{pred:.4f}" if pd.notna(pred) else "FAILED"
        actual_str = f"{actual:.4f}" if pd.notna(actual) else "N/A"
        print(f"  {vbin:12s} | Predicted: {pred_str:8s} | Actual: {actual_str}")

    # Summary stats
    print(f"\n--- Summary Statistics ---")
    if valid_mask.any():
        print(f"Predicted variance: min={summary_df['predicted_variance'].min():.4f}, max={summary_df['predicted_variance'].max():.4f}, mean={summary_df['predicted_variance'].mean():.4f}")
        print(f"Actual variance:    min={summary_df['ground_truth_variance'].min():.4f}, max={summary_df['ground_truth_variance'].max():.4f}, mean={summary_df['ground_truth_variance'].mean():.4f}")

    total_time = sum(r['elapsed_seconds'] for r in results)
    print(f"\n--- Complete ---")
    print(f"Analyzed {len(results)} markets in {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")

    return results, summary_df


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":

    results, summary = run_analysis(
        test_data_path='prediction_market_data/03_analysis_set.csv',
        output_dir='./variance_predictions_v9',
        max_markets=804
    )