!pip install evaluate datasets sacrebleu
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import os
import re
from tqdm import tqdm
import warnings
import ipywidgets as widgets
from IPython.display import display, clear_output
import json
from datetime import datetime
import wandb
wandb.login(key="b072fb2ed0b9f1df328fa056dfebb7dfadf71d6f")
warnings.filterwarnings("ignore")

# Import modules with try-except to handle different environments
try:
    from transformers.trainer_utils import get_last_checkpoint
except ImportError:
    from transformers.trainer_callback import TrainerState

try:
    import evaluate
except ImportError:
    evaluate = None
    print("Warning: 'evaluate' package not found. Will attempt to use sacrebleu directly if needed.")

# Configuration parameters (easy to modify)
# You can change these values directly in the notebook
CONFIG = {
    "data_file": "dataset.xlsx",  # Update this with your file path
    "model_name": "Helsinki-NLP/opus-mt-en-ur",
    "output_dir": "./fine-tuned-en-ur-model",
    "batch_size": 16,
    "learning_rate": 5e-5,
    "epochs": 3,
    "max_length": 128,
    "seed": 42,
    "eval_split": 0.1,
    "early_stopping_patience": 3,  # Number of evaluations with no improvement after which training will be stopped
    "test_examples": 500,  # Number of examples to test when using the Test button
    "bleu_output_file": "bleu_scores.json"  # File to save BLEU scores
}

# Global variables
model = None
tokenizer = None
split_dataset = None

def clean_text(text):
    """Clean and normalize text data"""
    if pd.isna(text):
        return ""

    # Convert to string if not already
    text = str(text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters that might interfere with training
    text = re.sub(r'[^\w\s\.\?\!\,\;\:\-\'\"\[\]\{\}\(\)،؟]', '', text)

    return text.strip()

def load_and_preprocess_data(file_path, eval_split=0.1):
    """
    Load the Excel file and preprocess the data
    """
    print(f"Loading data from {file_path}...")

    # Load the Excel file
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        print("Trying alternative Excel engines...")

        # Try multiple Excel engines
        for engine in ['openpyxl', 'xlrd', 'odf']:
            try:
                print(f"Attempting to load with {engine} engine...")
                df = pd.read_excel(file_path, engine=engine)
                print(f"Successfully loaded with {engine} engine")
                break
            except Exception as nested_e:
                print(f"Failed with {engine} engine: {nested_e}")
        else:
            raise ValueError("Could not load the Excel file with any available engine")

    # Print column names for debugging
    print(f"Columns found in file: {df.columns.tolist()}")

    # Check if required columns exist
    required_columns = ['English Sentence', 'Urdu Sentence']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        # Try to find columns with similar names
        for missing_col in missing_columns:
            base_name = missing_col.split()[0].lower()  # Get first word (English/Urdu)
            potential_matches = [col for col in df.columns if base_name.lower() in col.lower()]

            if potential_matches:
                print(f"Found potential matches for '{missing_col}': {potential_matches}")
                # Use the first match
                if missing_col == 'English Sentence':
                    df['English Sentence'] = df[potential_matches[0]]
                elif missing_col == 'Urdu Sentence':
                    df['Urdu Sentence'] = df[potential_matches[0]]

                print(f"Using '{potential_matches[0]}' as '{missing_col}'")

    # Create missing columns if they still don't exist (for graceful handling)
    for col in required_columns:
        if col not in df.columns:
            print(f"Creating empty column '{col}' as it was not found in the file")
            df[col] = ""

    # Extract only the required columns
    df = df[required_columns].copy()

    # Clean the text
    print("Cleaning text data...")
    df['English Sentence'] = df['English Sentence'].apply(clean_text)
    df['Urdu Sentence'] = df['Urdu Sentence'].apply(clean_text)

    # Remove empty rows
    df = df[(df['English Sentence'] != "") & (df['Urdu Sentence'] != "")]

    if len(df) == 0:
        raise ValueError("No valid translation pairs found after cleaning. Please check your file.")

    print(f"Dataset size after cleaning: {len(df)} pairs")

    # Convert to Hugging Face dataset
    dataset = Dataset.from_pandas(df)

    # Split into train and validation sets
    dataset = dataset.shuffle(seed=CONFIG["seed"])
    split_dataset = dataset.train_test_split(test_size=eval_split)

    return split_dataset

def preprocess_function(examples, tokenizer, max_length):
    """
    Tokenize the inputs and targets
    """
    source_texts = examples["English Sentence"]
    target_texts = examples["Urdu Sentence"]

    # Tokenize inputs
    model_inputs = tokenizer(
        source_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None  # Ensure we're not getting unexpected tensor types
    )

    # Tokenize targets
    # Different versions of transformers handle this differently
    try:
        # Newer versions
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                target_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )
    except (AttributeError, TypeError):
        # Older versions or different tokenizer types
        # Try direct tokenization for target with prefix
        prefix = ""
        if hasattr(tokenizer, 'lang_code_to_id'):
            # MBart-style tokenizer might need language code
            if 'ur' in tokenizer.lang_code_to_id:
                prefix = "ur_PK "  # Add Urdu language code prefix

        labels = tokenizer(
            [prefix + txt for txt in target_texts],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )

    model_inputs["labels"] = labels["input_ids"]

    # Replace pad token id with -100 in labels (only in HF trainer context)
    if tokenizer.pad_token_id is not None:
        model_inputs["labels"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
            for labels_example in model_inputs["labels"]
        ]

    return model_inputs

def compute_metrics(eval_preds):
    """
    Compute BLEU score for evaluation
    """
    try:
        # First try with the evaluate library
        bleu = evaluate.load("sacrebleu")

        preds, labels = eval_preds
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Convert ids to tokens
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # BLEU expects a list of translations for each reference
        formatted_labels = [[label] for label in decoded_labels]

        # Calculate BLEU score
        result = bleu.compute(predictions=decoded_preds, references=formatted_labels)

        return {"bleu": result["score"]}

    except (ImportError, ModuleNotFoundError):
        # Fallback to use sacrebleu directly if evaluate is not available
        try:
            import sacrebleu

            preds, labels = eval_preds
            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            # Convert ids to tokens
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # BLEU expects a list of translations for each reference
            formatted_labels = [[label] for label in decoded_labels]

            # Calculate BLEU score using sacrebleu directly
            bleu = sacrebleu.corpus_bleu(decoded_preds, formatted_labels)

            return {"bleu": bleu.score}

        except ImportError:
            print("Warning: Neither 'evaluate' nor 'sacrebleu' are available. Skipping BLEU calculation.")
            return {"bleu": 0.0}

def train_model_function(button):
    """
    Train the English-to-Urdu translation model
    """
    global model, tokenizer, split_dataset

    # Clear output for cleaner display
    clear_output(wait=True)

    # Disable the button during training
    button.disabled = True
    button.description = "Training..."

    try:
        # Set random seed for reproducibility
        torch.manual_seed(CONFIG["seed"])
        np.random.seed(CONFIG["seed"])

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Print versions for debugging
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Pandas version: {pd.__version__}")

        # Verify the data file exists
        if not os.path.exists(CONFIG["data_file"]):
            print(f"Warning: Data file '{CONFIG['data_file']}' not found.")
            CONFIG["data_file"] = input("Please enter the correct path to your Excel file: ")

        # Load the dataset
        split_dataset = load_and_preprocess_data(CONFIG["data_file"], CONFIG["eval_split"])
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(eval_dataset)}")

        # Load the pre-trained model and tokenizer
        print(f"Loading model: {CONFIG['model_name']}")
        model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

        # Preprocess the datasets
        print("Preprocessing datasets...")
        tokenized_train_dataset = train_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, CONFIG["max_length"]),
            batched=True,
            desc="Tokenizing train dataset",
        )

        tokenized_eval_dataset = eval_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, CONFIG["max_length"]),
            batched=True,
            desc="Tokenizing validation dataset",
        )

        # Define training arguments
        try:
            # Try with newer transformers version arguments
            training_args = Seq2SeqTrainingArguments(
                output_dir=CONFIG["output_dir"],
                per_device_train_batch_size=CONFIG["batch_size"],
                per_device_eval_batch_size=CONFIG["batch_size"],
                learning_rate=CONFIG["learning_rate"],
                num_train_epochs=CONFIG["epochs"],
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_bleu",
                greater_is_better=True,
                predict_with_generate=True,
                fp16=torch.cuda.is_available(),  # Use FP16 if available
                report_to="none",  # Disable Wandb and other reporting
            )
        except (TypeError, ValueError):
            # Fallback for older transformers version
            print("Using compatible training arguments for older transformers version")
            # The key issue: eval_strategy and save_strategy must match for load_best_model_at_end
            training_args = Seq2SeqTrainingArguments(
                output_dir=CONFIG["output_dir"],
                per_device_train_batch_size=CONFIG["batch_size"],
                per_device_eval_batch_size=CONFIG["batch_size"],
                learning_rate=CONFIG["learning_rate"],
                num_train_epochs=CONFIG["epochs"],
                eval_steps=500,
                save_steps=500,  # Must match eval_steps
                save_total_limit=2,
                # Don't use load_best_model_at_end in old versions to avoid conflicts
                predict_with_generate=True,
                fp16=torch.cuda.is_available(),  # Use FP16 if available
            )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="max_length",
            max_length=CONFIG["max_length"],
        )

        # Initialize the trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # Check if training was interrupted and can be resumed
        last_checkpoint = None
        if os.path.isdir(CONFIG["output_dir"]):
            try:
                last_checkpoint = get_last_checkpoint(CONFIG["output_dir"])
                if last_checkpoint:
                    print(f"Resuming training from checkpoint: {last_checkpoint}")
            except NameError:
                # Fallback for older transformers versions
                checkpoint_dirs = [os.path.join(CONFIG["output_dir"], d) for d in os.listdir(CONFIG["output_dir"])
                               if os.path.isdir(os.path.join(CONFIG["output_dir"], d)) and "checkpoint" in d]
                if checkpoint_dirs:
                    last_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
                    print(f"Resuming training from checkpoint: {last_checkpoint}")
            except Exception as e:
                print(f"Failed to find checkpoint: {e}")
                last_checkpoint = None

        # Start training
        print("Starting training...")
        try:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        except TypeError:
            # Older versions of transformers might not accept resume_from_checkpoint
            print("Warning: Could not resume from checkpoint with current transformers version.")
            trainer.train()

        # Save the final model
        print(f"Saving model to {CONFIG['output_dir']}")
        trainer.save_model(CONFIG["output_dir"])
        tokenizer.save_pretrained(CONFIG["output_dir"])

        # Evaluate on validation set
        print("Evaluating model...")
        results = trainer.evaluate()
        print(f"Evaluation results: {results}")

        print("\n✅ Training completed successfully!")

    except Exception as e:
        print(f"❌ An error occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting suggestions:")
        print("1. Check your file path and make sure the Excel file exists")
        print("2. Make sure the Excel file has the correct columns ('English Sentence' and 'Urdu Sentence')")
        print("3. Try installing additional packages: pip install openpyxl xlrd odfpy")

    # Re-enable the button
    button.disabled = False
    button.description = "Train Model"

def translate_text(text, model, tokenizer, max_length=128):
    """
    Translate English text to Urdu using the fine-tuned model.
    """
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    # Move to GPU if available
    if torch.cuda.is_available():
        model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=4)

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


def test_model_function(button):
    """
    Test the model and generate BLEU score file
    """
    global model, tokenizer, split_dataset

    # Clear output for cleaner display
    clear_output(wait=True)

    # Disable the button during testing
    button.disabled = True
    button.description = "Testing..."

    try:
        # Check if model and tokenizer are loaded
        if model is None or tokenizer is None:
            print("Loading model and tokenizer from saved directory...")
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["output_dir"])
                tokenizer = AutoTokenizer.from_pretrained(CONFIG["output_dir"])
            except Exception as e:
                print(f"❌ Failed to load model: {str(e)}")
                print("Please train the model first or provide a valid model directory.")
                button.disabled = False
                button.description = "Test Model"
                return

        # Load the dataset if not already loaded
        if split_dataset is None:
            split_dataset = load_and_preprocess_data(CONFIG["data_file"], CONFIG["eval_split"])

        # Get the test dataset
        test_dataset = split_dataset["test"]

        # Select examples for testing
        num_examples = min(CONFIG["test_examples"], len(test_dataset))
        examples = test_dataset.select(range(num_examples))

        # Initialize variables for calculating BLEU
        all_references = []
        all_predictions = []

        print(f"Testing model on {num_examples} examples:")

        # Process examples
        for i, example in enumerate(tqdm(examples, desc="Testing")):
            english_text = example["English Sentence"]
            reference_urdu = example["Urdu Sentence"]

            # Translate
            translated_urdu = translate_text(english_text, model, tokenizer)

            # Store for BLEU calculation
            all_references.append([reference_urdu])
            all_predictions.append(translated_urdu)

            # Print some examples (limit to first 5)
            if i < 5:
                print(f"\nExample {i+1}:")
                print(f"English: {english_text}")
                print(f"Reference Urdu: {reference_urdu}")
                print(f"Translated Urdu: {translated_urdu}")

        # Calculate BLEU score
        try:
            if evaluate:
                bleu = evaluate.load("sacrebleu")
                bleu_result = bleu.compute(predictions=all_predictions, references=all_references)
                bleu_score = bleu_result["score"]
            else:
                import sacrebleu
                bleu_result = sacrebleu.corpus_bleu(all_predictions, all_references)
                bleu_score = bleu_result.score
        except Exception as e:
            print(f"Warning: Error calculating BLEU score: {str(e)}")
            bleu_score = 0.0

        print(f"\nFinal BLEU score on {num_examples} test examples: {bleu_score:.2f}")

        # Save BLEU score to file
        bleu_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": CONFIG["model_name"],
            "num_test_examples": num_examples,
            "bleu_score": bleu_score,
            "examples": [
                {
                    "english": example["English Sentence"],
                    "reference_urdu": example["Urdu Sentence"],
                    "translated_urdu": translate_text(example["English Sentence"], model, tokenizer)
                }
                for example in examples.select(range(min(5, num_examples)))
            ]
        }

        with open(CONFIG["bleu_output_file"], "w", encoding="utf-8") as f:
            json.dump(bleu_data, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Testing completed successfully! BLEU scores saved to {CONFIG['bleu_output_file']}")

    except Exception as e:
        print(f"❌ An error occurred during testing: {str(e)}")
        import traceback
        traceback.print_exc()

    # Re-enable the button
    button.disabled = False
    button.description = "Test Model"

def initialize_widgets():
    """
    Create and display the interactive widgets for training and testing
    """
    # Create buttons
    train_button = widgets.Button(
        description="Train Model",
        button_style="primary",
        tooltip="Train the English-to-Urdu translation model",
        icon="graduation-cap"
    )

    test_button = widgets.Button(
        description="Test Model",
        button_style="success",
        tooltip="Test the model and generate BLEU scores",
        icon="check"
    )

    # Connect buttons to functions
    train_button.on_click(train_model_function)
    test_button.on_click(test_model_function)

    # Create a configuration output
    config_output = widgets.Output()
    with config_output:
        print("Current Configuration:")
        for key, value in CONFIG.items():
            print(f"  {key}: {value}")

    # Create tabs for the interface
    config_tab = widgets.VBox([widgets.HTML("<h3>Configuration</h3>"), config_output])
    buttons_tab = widgets.VBox([
        widgets.HTML("<h3>Model Operations</h3>"),
        widgets.HBox([train_button, test_button])
    ])

    # Create tab layout
    tabs = widgets.Tab(children=[buttons_tab, config_tab])
    tabs.set_title(0, "Operations")
    tabs.set_title(1, "Configuration")

    # Display the tabs
    display(tabs)

# Create and display the widgets when the notebook is run
if __name__ == "__main__" or 'ipykernel' in sys.modules:
    print("English-to-Urdu Translation Model")
    print("=================================")
    print("Use the 'Train Model' button to start training.")
    print("Use the 'Test Model' button to test the model and generate BLEU scores.")

    initialize_widgets()
