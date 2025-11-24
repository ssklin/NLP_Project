import os
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import Counter
from constants import OUTPUT_DIR, MODEL_DIR

MODEL_PATH = MODEL_DIR
TARGET_FILE = os.path.join(OUTPUT_DIR, "YGF Malatang_keywords.json")

MAX_KEYWORDS_PER_ASPECT = 10

print(f"Loading model from {MODEL_PATH}...")
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

    device = "cpu"
    model = model.to(device)
    print("Model loaded successfully!")
except OSError:
    print("Error: Model not found! Please run the training script first.")
    exit()


def prepare_input_from_json(file_path):
    """
    Reads the extracted keywords JSON and formats it for the model.
    Aggregates keywords by frequency.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    restaurant_name = data.get("Restaurant", "Restaurant")
    aspect_analysis = data.get("Aspect_Analysis", {})

    prompt_parts = []

    print(f"\nAggregating keywords for {restaurant_name}...")

    for aspect, sentiments in aspect_analysis.items():
        # Collect all keywords for this aspect (distinguishing Positive vs Negative if desired)

        all_kws = []
        for sentiment, entries in sentiments.items():
            for entry in entries:
                kw = entry.get("Keyword")
                if kw:
                    all_kws.append(kw.lower())

        if not all_kws:
            continue

        # Get Top N frequent keywords
        counter = Counter(all_kws)
        top_kws = [k for k, v in counter.most_common(MAX_KEYWORDS_PER_ASPECT)]

        if top_kws:
            prompt_parts.append(f"{aspect}: {', '.join(top_kws)}")

    # Final Prompt
    input_text = "summarize: " + " | ".join(prompt_parts)
    return input_text


def generate_summary(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    outputs = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True,
        repetition_penalty=2.5,
        no_repeat_ngram_size=3,
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


if __name__ == "__main__":
    print("Preparing input...")
    input_prompt = prepare_input_from_json(TARGET_FILE)

    if input_prompt:
        print(f"\nInput Prompt to Model:\n{input_prompt}")

        print("\nGenerating Summary...")
        summary = generate_summary(input_prompt)

        print("\n" + "=" * 30)
        print("FINAL GENERATED SUMMARY")
        print("=" * 30)
        print(summary)
        print("=" * 30)
