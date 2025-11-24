import os
import json
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import random
import re

# local files
from keyword_utils import load_nlp_model, extract_aspect_keywords, format_keywords_for_training
from constants import MODEL_DIR, YELP_DATA_PATH

# Colab
# !pip install transformers datasets sentencepiece accelerate spacy
# !python -m spacy download en_core_web_sm

# Disable WandB logging to avoid login prompt
os.environ["WANDB_DISABLED"] = "true"

# Paths
YELP_FILE = YELP_DATA_PATH

# Model Settings
MODEL_NAME = "t5-small"  # 't5-base' or 't5-small'
BATCH_SIZE = 2
EPOCHS = 1
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 256
SAMPLE_SIZE = 10000

nlp = load_nlp_model()


def convert_to_third_person(text):
    """
    Simple heuristic to convert first-person reviews to third-person summaries.
    """
    # 1. Replace "I", "We" with "Customers" or "The reviewer"
    # Use "Customers" for plural/general feel
    text = re.sub(r"\bI\b", "Customers", text, flags=re.IGNORECASE)
    text = re.sub(r"\bWe\b", "Customers", text, flags=re.IGNORECASE)

    # 2. Replace possessives
    text = re.sub(r"\bmy\b", "their", text, flags=re.IGNORECASE)
    text = re.sub(r"\bour\b", "their", text, flags=re.IGNORECASE)

    # 3. Replace objects
    text = re.sub(r"\bme\b", "them", text, flags=re.IGNORECASE)
    text = re.sub(r"\bus\b", "them", text, flags=re.IGNORECASE)

    # 4. Fix grammar slightly (very basic)
    # "Customers am" -> "Customers are" (rare but possible if "I am")
    text = re.sub(r"\bCustomers am\b", "Customers are", text)
    text = re.sub(r"\bCustomers was\b", "Customers were", text)

    return text


def load_and_process_data(filepath, limit=10000):
    print(f"Loading data from {filepath}...")
    data = []
    count = 0

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if count >= limit:
                    break
                review = json.loads(line)
                text = review.get("text", "").replace("\n", " ").strip()

                # Skip short reviews
                if len(text.split()) < 10:
                    continue

                # Extract keywords (Input)
                extracted = extract_aspect_keywords(text, nlp)
                keywords_str = format_keywords_for_training(extracted)

                # Only keep if we found keywords (meaningful input)
                if keywords_str:
                    # Convert Target to Third Person
                    target_text = convert_to_third_person(text)

                    data.append({"input_text": keywords_str, "target_text": target_text})
                    count += 1
                    if count % 1000 == 0:
                        print(f"Processed {count} reviews...")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []

    print(f"Total training samples: {len(data)}")
    return data


class ReviewSummaryDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len, max_target_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        input_text = "summarize: " + item["input_text"]
        target_text = item["target_text"]

        input_encoding = self.tokenizer(
            input_text, max_length=self.max_input_len, padding="max_length", truncation=True, return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text, max_length=self.max_target_len, padding="max_length", truncation=True, return_tensors="pt"
        )

        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss

        return {
            "input_ids": input_encoding.input_ids.flatten(),
            "attention_mask": input_encoding.attention_mask.flatten(),
            "labels": labels.flatten(),
        }


def get_model_and_tokenizer(model_name=MODEL_NAME):
    """
    Loads the model and tokenizer.
    """
    print(f"Loading Model: {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer


def prepare_datasets(tokenizer, filepath=YELP_FILE, limit=SAMPLE_SIZE):
    """
    Loads data, splits it, and creates Dataset objects.
    """
    # 1. Load Data
    raw_data = load_and_process_data(filepath, limit=limit)
    if not raw_data:
        return None, None

    # Split Train/Val
    random.shuffle(raw_data)
    split = int(len(raw_data) * 0.9)
    train_data = raw_data[:split]
    val_data = raw_data[split:]

    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # 3. Datasets
    train_dataset = ReviewSummaryDataset(train_data, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    val_dataset = ReviewSummaryDataset(val_data, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)

    return train_dataset, val_dataset


def run_training(model, tokenizer, train_dataset, val_dataset, output_dir=MODEL_DIR):
    """
    Sets up the Trainer and runs training.
    """
    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=False,  # Force FP32 for CPU
        no_cuda=True,  # Force CPU
        report_to="none",
    )

    # 5. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # 6. Train
    print("Starting Training...")
    trainer.train()

    # 7. Save
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")
    return trainer


if __name__ == "__main__":
    device = "cpu"
    print(f"Using device: {device}")

    # Step 1: Load Model
    model, tokenizer = get_model_and_tokenizer()

    # Step 2: Prepare Data
    train_dataset, val_dataset = prepare_datasets(tokenizer)

    # Step 3: Train
    if train_dataset and val_dataset:
        run_training(model, tokenizer, train_dataset, val_dataset)
