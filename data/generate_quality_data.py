"""
Generate quality training data using GPT-5-mini.
Creates 100 high-quality and 100 low-quality argument samples
for training the quality evaluation model.
"""

import os
import csv
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")

# Initialize OpenAI client
client = OpenAI()

# One-shot examples - these are extracted arguments (claims + evidence), not full abstracts
HIGH_QUALITY_EXAMPLE = """The telephone-delivered health coaching intervention significantly improved physical activity levels in cancer survivors. At 12 months, participants showed a 28.5 minute increase in moderate physical activity (P = .003). Body mass index decreased by 0.9 kg/m² (P = .001). Total fat intake reduced by 7.0% (P = .006) and saturated fat by 2.8% (P = .016). These findings demonstrate that structured behavioral interventions can effectively modify multiple health behaviors simultaneously in at-risk populations."""

LOW_QUALITY_EXAMPLE = """The proposed technology has the potential to enhance results significantly. This approach combines Internet of Things with advanced algorithms to improve performance. The model has been optimized during implementation. Compared with traditional methods, the processing speed is improved. The system design incorporates multiple technologies to address existing issues. Results verified the applicability and efficiency of the proposed model."""


def generate_sample_gpt5(quality_type: str, topic_hint: str = "") -> str:
    """Generate a single quality sample using GPT-5-mini with Responses API."""
    
    if quality_type == "high":
        example = HIGH_QUALITY_EXAMPLE
        quality_desc = "HIGH quality"
        characteristics = """
- Specific quantitative results with statistical significance (p-values, confidence intervals)
- Concrete claims directly supported by numerical evidence
- Clear cause-and-effect relationships
- Precise measurements and sample sizes
- Professional scientific language with specific terminology"""
    else:
        example = LOW_QUALITY_EXAMPLE
        quality_desc = "LOW quality"
        characteristics = """
- Vague claims without specific numbers or statistics
- Generic statements like "results show improvement" without data
- Buzzwords without substance (e.g., "potential", "enhanced", "optimized")
- Circular reasoning or tautologies
- Missing methodology details
- No statistical significance or confidence measures"""

    prompt = f"""Generate an extracted scientific argument that represents {quality_desc} argumentation.

An extracted argument consists of claims and their supporting evidence, as might be extracted from a scientific abstract by an argument mining system. This is NOT a full abstract - it's the key claims and evidence only, typically 50-100 words.

Characteristics of {quality_desc} arguments:
{characteristics}

Example of {quality_desc} extracted argument:
"{example}"

Now generate a NEW {quality_desc} extracted argument on a different scientific topic{' related to ' + topic_hint if topic_hint else ''}. 
Keep it to 50-100 words - just the claims and evidence, not a full abstract. Only output the argument text, nothing else."""

    system_message = "You are a scientific writing expert who can generate both high-quality and low-quality scientific abstracts for training purposes."
    
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
        max_output_tokens=2000
    )
    
    # Check response status
    if hasattr(response, "status") and response.status == "incomplete":
        reason = getattr(response.incomplete_details, "reason", "unknown") if hasattr(response, "incomplete_details") else "unknown"
        raise RuntimeError(f"GPT-5-mini returned incomplete response. Reason: {reason}")
    
    result = getattr(response, "output_text", None)
    if not result:
        raise RuntimeError("GPT-5-mini returned empty output_text")
    
    return result.strip()


def main():
    # Topics to generate diverse samples
    topics = [
        "machine learning", "cancer treatment", "climate change", "neuroscience",
        "drug discovery", "renewable energy", "genetics", "psychology",
        "materials science", "epidemiology", "robotics", "nutrition",
        "cardiovascular health", "immunology", "environmental science",
        "computer vision", "pharmacology", "ecology", "nanotechnology",
        "cognitive science"
    ]
    
    output_file = "quality_training_data.csv"
    samples = []
    
    print("Generating HIGH quality samples using GPT-5-mini...")
    for i in tqdm(range(100)):
        topic = topics[i % len(topics)]
        try:
            text = generate_sample_gpt5("high", topic)
            samples.append((text, 1))  # 1 = high quality
        except Exception as e:
            print(f"Error generating high quality sample {i}: {e}")
            continue
    
    print("\nGenerating LOW quality samples using GPT-5-mini...")
    for i in tqdm(range(100)):
        topic = topics[i % len(topics)]
        try:
            text = generate_sample_gpt5("low", topic)
            samples.append((text, 0))  # 0 = low quality
        except Exception as e:
            print(f"Error generating low quality sample {i}: {e}")
            continue
    
    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        for text, label in samples:
            writer.writerow([text, label])
    
    print(f"\n✓ Generated {len(samples)} samples")
    print(f"✓ Saved to {output_file}")
    print(f"  - High quality (label=1): {sum(1 for _, l in samples if l == 1)}")
    print(f"  - Low quality (label=0): {sum(1 for _, l in samples if l == 0)}")


if __name__ == "__main__":
    main()
