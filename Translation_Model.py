
# Install required packages
!pip install evaluate datasets sacrebleu pandas torch transformers openpyxl ipywidgets scikit-learn

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
from tqdm import tqdm
import warnings
import ipywidgets as widgets
from IPython.display import display, clear_output
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

try:
    import evaluate
except ImportError:
    evaluate = None
    print("Warning: 'evaluate' not found. Using sacrebleu as fallback.")

# Configuration
CONFIG = {
    "data_file": "All_Drama_Urdu_Eng.xlsx",
    "model_name": "Helsinki-NLP/opus-mt-en-ur",
    "output_dir": "./fine-tuned-en-ur-model",
    "batch_size": 8,  # Reduced for faster training
    "learning_rate": 5e-5,
    "epochs": 1,  # Reduced for speed
    "max_length": 64,  # Reduced for efficiency
    "seed": 42,
    "eval_split": 0.1,
    "max_train_samples": 1000,  # Subset for quick training
    "test_examples": 50,  # Reduced for quick testing
    "bleu_output_file": "bleu_scores.json"
}

# Global variables
model = None
tokenizer = None
split_dataset = None

def clean_text(text):
    """Clean text data"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    return text

def load_and_preprocess_data(file_path, eval_split=0.1, max_samples=None):
    """Load and preprocess Excel data"""
    print(f"Loading {file_path}...")
    if not os.path.exists(file_path):
        file_path = input("File not found. Enter Excel file path: ")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"Error loading Excel: {e}")
        raise ValueError("Failed to load Excel file.")

    print(f"Columns: {df.columns.tolist()}")
    print("First 3 rows:\n", df.head(3))

    eng_col = 'english sentence'
    urdu_col = 'urdu sentence'
    label_col = 'label'
    if eng_col not in df.columns or urdu_col not in df.columns:
        raise ValueError("Required columns 'english sentence' and 'urdu sentence' not found.")

    df = df[[eng_col, urdu_col, label_col]].copy()
    df.columns = ['English Sentence', 'Urdu Sentence', 'Label']
    df['English Sentence'] = df['English Sentence'].apply(clean_text)
    df['Urdu Sentence'] = df['Urdu Sentence'].apply(clean_text)

    df = df[df['English Sentence'] != ""]
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=CONFIG["seed"])
    if len(df) == 0:
        raise ValueError("No valid data after cleaning.")

    print(f"Dataset size: {len(df)} pairs")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed=CONFIG["seed"])
    return dataset.train_test_split(test_size=eval_split)

def preprocess_function(examples, tokenizer, max_length):
    """Tokenize inputs and targets"""
    source_texts = examples["English Sentence"]
    target_texts = examples["Urdu Sentence"]
    model_inputs = tokenizer(source_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors=None)

    try:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(target_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors=None)
    except:
        prefix = "ur_PK " if hasattr(tokenizer, 'lang_code_to_id') and 'ur' in tokenizer.lang_code_to_id else ""
        labels = tokenizer([prefix + txt for txt in target_texts], padding="max_length", truncation=True, max_length=max_length, return_tensors=None)

    model_inputs["labels"] = [[(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
                             for labels_example in labels["input_ids"]]
    return model_inputs

def compute_bleu(predictions, references):
    """Compute BLEU score"""
    try:
        bleu = evaluate.load("sacrebleu") if evaluate else None
        formatted_refs = [[ref] for ref in references]
        if evaluate:
            result = bleu.compute(predictions=predictions, references=formatted_refs)
            return result["score"]
        else:
            import sacrebleu
            return sacrebleu.corpus_bleu(predictions, formatted_refs).score
    except Exception as e:
        print(f"Warning: BLEU calculation failed: {e}")
        return 0.0

def compute_classification_metrics(predictions, references):
    """Compute accuracy, precision, recall, F1"""
    accuracy = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, zero_division=0)
    recall = recall_score(references, predictions, zero_division=0)
    f1 = f1_score(references, predictions, zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def train_model(button):
    """Train the model"""
    global model, tokenizer, split_dataset
    clear_output(wait=True)
    button.disabled = True
    button.description = "Training..."

    try:
        torch.manual_seed(CONFIG["seed"])
        np.random.seed(CONFIG["seed"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        split_dataset = load_and_preprocess_data(CONFIG["data_file"], CONFIG["eval_split"], CONFIG["max_train_samples"])
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(eval_dataset)}")

        model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

        tokenized_train_dataset = train_dataset.map(
            lambda x: preprocess_function(x, tokenizer, CONFIG["max_length"]), batched=True)
        tokenized_eval_dataset = eval_dataset.map(
            lambda x: preprocess_function(x, tokenizer, CONFIG["max_length"]), batched=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=CONFIG["output_dir"],
            per_device_train_batch_size=CONFIG["batch_size"],
            per_device_eval_batch_size=CONFIG["batch_size"],
            learning_rate=CONFIG["learning_rate"],
            num_train_epochs=CONFIG["epochs"],
            eval_steps=100,
            save_steps=100,
            save_total_limit=1,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="max_length", max_length=CONFIG["max_length"])
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=lambda x: {"bleu": compute_bleu(x[0], [ex["Urdu Sentence"] for ex in eval_dataset])},
        )

        trainer.train()
        trainer.save_model(CONFIG["output_dir"])
        tokenizer.save_pretrained(CONFIG["output_dir"])

        results = trainer.evaluate()
        print(f"Evaluation results: BLEU = {results.get('eval_bleu', 0.0):.2f}")
        print("\n✅ Training completed!")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Troubleshooting: Check Excel file path, column names, and package installations.")

    button.disabled = False
    button.description = "Train Model"

def translate_text(text, model, tokenizer, max_length=64):
    """Translate English to Urdu"""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    if torch.cuda.is_available():
        model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def test_model(button):
    """Test model with user input and compute metrics"""
    global model, tokenizer, split_dataset
    clear_output(wait=True)
    button.disabled = True
    button.description = "Testing..."

    try:
        if model is None or tokenizer is None:
            model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["output_dir"])
            tokenizer = AutoTokenizer.from_pretrained(CONFIG["output_dir"])

        if split_dataset is None:
            split_dataset = load_and_preprocess_data(CONFIG["data_file"], CONFIG["eval_split"], CONFIG["max_train_samples"])

        # Prompt for user input
        print("Enter an English sentence to translate to Urdu (or press Enter to skip):")
        eng_input = input().strip()
        print("Enter the expected Urdu translation (or press Enter to skip):")
        urdu_ref = input().strip()

        predictions, references, labels = [], [], []
        if eng_input and urdu_ref:
            translated = translate_text(eng_input, model, tokenizer)
            print(f"\nUser Input Test:")
            print(f"English: {eng_input}")
            print(f"Reference Urdu: {urdu_ref}")
            print(f"Translated Urdu: {translated}")

            # Assume translation is correct if it matches reference (label=1)
            predictions.append(1 if translated.strip() == urdu_ref.strip() else 0)
            labels.append(1)  # User expects correct translation
            references.append(urdu_ref)
        else:
            print("No user input provided. Testing on dataset samples.")

        # Test on dataset samples
        test_dataset = split_dataset["test"]
        num_examples = min(CONFIG["test_examples"], len(test_dataset))
        examples = test_dataset.select(range(num_examples))

        print(f"\nTesting on {num_examples} dataset examples:")
        for i, example in enumerate(tqdm(examples, desc="Testing")):
            english_text = example["English Sentence"]
            reference_urdu = example["Urdu Sentence"]
            label = example["Label"]
            translated_urdu = translate_text(english_text, model, tokenizer)
            predictions.append(1 if translated_urdu.strip() == reference_urdu.strip() else 0)
            labels.append(label)
            references.append(reference_urdu)
            if i < 3:
                print(f"\nExample {i+1}:")
                print(f"English: {english_text}")
                print(f"Reference Urdu: {reference_urdu}")
                print(f"Translated Urdu: {translated_urdu}")

        # Compute metrics
        bleu_score = compute_bleu(predictions, references)
        metrics = compute_classification_metrics(predictions, labels)

        print(f"\nResults:")
        print(f"BLEU Score: {bleu_score:.2f}")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"F1 Score: {metrics['f1']:.2f}")

        # Save results
        results_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": CONFIG["model_name"],
            "num_examples": len(predictions),
            "bleu_score": bleu_score,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "user_test": {
                "english": eng_input,
                "reference_urdu": urdu_ref,
                "translated_urdu": translated if eng_input and urdu_ref else None
            } if eng_input and urdu_ref else None
        }
        with open(CONFIG["bleu_output_file"], "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {CONFIG['bleu_output_file']}")

    except Exception as e:
        print(f"❌ Error during testing: {e}")

    button.disabled = False
    button.description = "Test Model"

def interactive_widgets():
    """Create interactive widgets"""
    train_button = widgets.Button(description="Train Model", button_style="primary", tooltip="Train model", icon="graduation-cap")
    test_button = widgets.Button(description="Test Model", button_style="success", tooltip="Test model", icon="check")
    train_button.on_click(train_model)
    test_button.on_click(test_model)
    display(widgets.HBox([train_button, test_button]))

if __name__ == "__main__" or "ipykernel" in sys.modules:
    print("English-to-Urdu Translation Model")
    print("Use 'Train Model' to train, 'Test Model' to test with input.")
    interactive_widgets()
