"""
GPT-5-mini One-Shot Classifier for Fraudulent vs Legitimate Scientific Abstracts

This script uses GPT-5-mini with one-shot examples to classify abstracts
as either fraudulent (low quality) or legitimate (high quality).
"""

import os
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")

# Initialize OpenAI client
client = OpenAI()


def load_abstracts(file_path: str) -> list[dict]:
    """Load abstracts from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    abstracts = []
    for doc_id, content in data.items():
        sentences = content.get('sentences', [])
        # Join sentences into full abstract, filtering empty strings
        text = ' '.join([s.strip() for s in sentences if s and s.strip()])
        if text:
            abstracts.append({
                'id': doc_id,
                'text': text
            })
    return abstracts


def get_one_shot_examples(legitimate_abstracts: list[dict], fraudulent_abstracts: list[dict]) -> tuple[str, str]:
    """Get one example from each dataset for one-shot learning."""
    # Use the first abstract from each dataset as examples
    legitimate_example = legitimate_abstracts[0]['text'][:1500]  # Truncate if too long
    fraudulent_example = fraudulent_abstracts[0]['text'][:1500]
    return legitimate_example, fraudulent_example


def classify_abstract_gpt5(abstract_text: str, legitimate_example: str, fraudulent_example: str) -> int:
    """
    Classify an abstract using GPT-5-mini with one-shot examples.
    Returns: 1 for legitimate (high quality), 0 for fraudulent (low quality)
    """
    
    prompt = f"""You are an expert at detecting fraudulent or low-quality scientific abstracts. You must interpret low-quality very broadly and not be hesitant to classify an abstract as fraudulent if you think there are indication of lower quality. Following are examples of a legitimate and a fraudulent abstract:

LEGITIMATE (High Quality) Example:
"{legitimate_example}"

FRAUDULENT (Low Quality) Example:
"{fraudulent_example}"

Now classify this abstract:
"{abstract_text[:2000]}"

Respond with ONLY one word: either "LEGITIMATE" or "FRAUDULENT"."""

    system_message = "You are an expert scientific paper reviewer who can distinguish between legitimate high-quality research and fraudulent or low-quality papers."
    
    # Use GPT-5 Responses API
    input_messages = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_message}]
        },
        {
            "role": "user", 
            "content": [{"type": "input_text", "text": prompt}]
        }
    ]
    
    response = client.responses.create(
        model="gpt-5-mini",
        input=input_messages,
        max_output_tokens=1000
    )
    
    # Check response status
    if hasattr(response, "status") and response.status == "incomplete":
        raise RuntimeError("GPT-5-mini returned incomplete response")
    
    result = getattr(response, "output_text", "")
    if not result:
        raise RuntimeError("GPT-5-mini returned empty output")
    
    # Parse response
    result = result.strip().upper()
    if "LEGITIMATE" in result:
        return 1
    elif "FRAUDULENT" in result:
        return 0
    else:
        # Default to fraudulent if unclear
        print(f"  Warning: Unclear response '{result}', defaulting to fraudulent")
        return 0


def main():
    # File paths
    legitimate_file = "legitimate_abstracts.json"
    fraudulent_file = "fraudulent_val.json"
    
    print("Loading datasets...")
    legitimate_abstracts = load_abstracts(legitimate_file)
    fraudulent_abstracts = load_abstracts(fraudulent_file)
    
    print(f"  Legitimate abstracts: {len(legitimate_abstracts)}")
    print(f"  Fraudulent abstracts: {len(fraudulent_abstracts)}")
    
    # Get one-shot examples (skip the first one from each dataset since we'll use it as example)
    legitimate_example, fraudulent_example = get_one_shot_examples(
        legitimate_abstracts, fraudulent_abstracts
    )
    
    # Use remaining abstracts for testing (skip first which is used as example)
    test_legitimate = legitimate_abstracts[1:]
    test_fraudulent = fraudulent_abstracts[1:]
    
    # Limit samples for cost/time (optional - remove or increase for full evaluation)
    MAX_SAMPLES_PER_CLASS = 50
    test_legitimate = test_legitimate[:MAX_SAMPLES_PER_CLASS]
    test_fraudulent = test_fraudulent[:MAX_SAMPLES_PER_CLASS]
    
    print(f"\nClassifying {len(test_legitimate)} legitimate and {len(test_fraudulent)} fraudulent abstracts...")
    print(f"Using GPT-5-mini with one-shot learning\n")
    
    # Classify legitimate abstracts
    y_true = []
    y_pred = []
    
    print("Classifying LEGITIMATE abstracts...")
    for abstract in tqdm(test_legitimate):
        try:
            pred = classify_abstract_gpt5(abstract['text'], legitimate_example, fraudulent_example)
            y_pred.append(pred)
            y_true.append(1)  # True label is legitimate
        except Exception as e:
            print(f"  Error classifying {abstract['id']}: {e}")
            continue
    
    print("\nClassifying FRAUDULENT abstracts...")
    for abstract in tqdm(test_fraudulent):
        try:
            pred = classify_abstract_gpt5(abstract['text'], legitimate_example, fraudulent_example)
            y_pred.append(pred)
            y_true.append(0)  # True label is fraudulent
        except Exception as e:
            print(f"  Error classifying {abstract['id']}: {e}")
            continue
    
    # Calculate metrics
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS (GPT-5-mini One-Shot)")
    print("="*60)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    print(f"\nTotal samples classified: {len(y_true)}")
    print(f"  - Legitimate samples: {sum(1 for y in y_true if y == 1)}")
    print(f"  - Fraudulent samples: {sum(1 for y in y_true if y == 0)}")
    
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Fraud   Legit")
    print(f"  Actual Fraud    {tn:4d}    {fp:4d}")
    print(f"  Actual Legit    {fn:4d}    {tp:4d}")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (of predicted legitimate, how many are correct)")
    print(f"  Recall:    {recall:.4f} (of actual legitimate, how many were found)")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Per-class accuracy
    legit_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    fraud_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nPer-class accuracy:")
    print(f"  Legitimate detection: {legit_acc:.4f} ({tp}/{tp+fn})")
    print(f"  Fraudulent detection: {fraud_acc:.4f} ({tn}/{tn+fp})")
    
    print("\n" + "="*60)
    print("Full Classification Report:")
    print("="*60)
    print(classification_report(y_true, y_pred, 
                                target_names=['Fraudulent', 'Legitimate'],
                                zero_division=0))


if __name__ == "__main__":
    main()

