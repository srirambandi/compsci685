import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
import numpy as np
import re
from typing import Dict, List, Tuple, Optional


def load_dataset(file_path='../gen_dataset/data/fin_dataset.csv'):


    # Load the dataset
    try:
        # Try reading with header first
        try:
            df = pd.read_csv(file_path)
            if 'equ_prefix' not in df.columns or 'sol_prefix' not in df.columns:
                # If expected columns missing, try without header
                raise FileNotFoundError # Trigger the next try block cleanly
        except (FileNotFoundError, pd.errors.EmptyDataError, Exception): # Broad catch if header guess fails
            print("Could not read with header or file issue, attempting without header...")
            # Attempt to load assuming no header
            # Example: id,equ_str,equ_prefix,sol_str,sol_prefix
            # Indices:   0,      1,         2,       3,         4
            df = pd.read_csv(file_path, header=None)
            # Assign default names based on the order in the example snippet
            if df.shape[1] >= 5: # Check if enough columns exist based on example
                # Adjust column names based on the structure you provided
                df.columns = ['id', 'equ_str', 'equ_prefix', 'sol_str', 'sol_prefix'] + [f'extra_{i}' for i in range(df.shape[1] - 5)]
                print("Read CSV without header, assigned default column names.")
            else:
                # If still not enough columns, raise error
                raise ValueError(f"CSV file '{file_path}' does not contain the expected 5+ columns to infer 'equ_prefix' and 'sol_prefix'. Shape: {df.shape}")

        # Select only the prefix columns needed for input/output, plus sol_str for evaluation
        if 'equ_prefix' in df.columns and 'sol_prefix' in df.columns and 'sol_str' in df.columns:
            df = df[['equ_prefix', 'sol_prefix', 'sol_str']] # Keep sol_str
        elif 'equ_prefix' in df.columns and 'sol_prefix' in df.columns:
            print("Warning: 'sol_str' column not found, SymPy evaluation requiring infix ground truth will not be possible.")
            df = df[['equ_prefix', 'sol_prefix']]
        else:
            # This case should be caught by earlier checks, but as a safeguard:
            raise ValueError("Could not identify 'equ_prefix' and 'sol_prefix' columns after loading.")


        # Remove rows with missing values in essential columns
        df.dropna(subset=['equ_prefix', 'sol_prefix'], inplace=True)
        # Ensure columns are strings
        df['equ_prefix'] = df['equ_prefix'].astype(str)
        df['sol_prefix'] = df['sol_prefix'].astype(str)
        if 'sol_str' in df.columns:
            df['sol_str'] = df['sol_str'].astype(str)


        # Convert DataFrame to Hugging Face Dataset
        raw_dataset = Dataset.from_pandas(df)

        # Basic train/test split (adjust ratio as needed)
        # Ensure there's enough data for the split
        if len(raw_dataset) < 2:
            print("Warning: Dataset has less than 2 samples, cannot create test split. Using all data for training.")
            # Create an empty test dataset with the same features if needed downstream
            empty_test_dict = {k: [] for k in raw_dataset.features.keys()}
            test_dataset = Dataset.from_dict(raw_dataset.features.deserialize(raw_dataset.features.serialize(empty_test_dict)))
            dataset = DatasetDict({
                'train': raw_dataset,
                'test': test_dataset
            })
        else:
            # Ensure test_size is at least 1 example if possible, max 10%
            test_size = max(1/len(raw_dataset), 0.1) if len(raw_dataset) >= 10 else 1/len(raw_dataset)
            train_test_split = raw_dataset.train_test_split(test_size=test_size, seed=42) # Add seed for reproducibility
            dataset = DatasetDict({
                'train': train_test_split['train'],
                'test': train_test_split['test']
            })


        print("Dataset loaded and split:")
        print(dataset)
        # Print features to confirm 'sol_str' presence
        print("\nDataset features:")
        print(dataset['train'].features)


    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        dataset = None
    except ValueError as e:
        print(f"Error loading or processing CSV: {e}")
        dataset = None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        dataset = None

    # --- Model and Tokenizer Setup ---
    # Ensure transformers and torch are imported (already done above)
    model_checkpoint = "t5-small" # Or another small encoder-decoder model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # --- Tokenization Function ---
    max_input_length = 128
    max_target_length = 128

    def preprocess_function(examples):
        # Add prefix if needed by the model (e.g., T5 requires a task prefix)
        task_prefix = "translate prefix equation to prefix solution: "
        inputs = [task_prefix + str(eq) for eq in examples["equ_prefix"]] # Ensure input is string
        targets = [str(sol) for sol in examples["sol_prefix"]] # Ensure target is string

        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=False) # Padding handled by collator

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding=False) # Padding handled by collator

        model_inputs["labels"] = labels["input_ids"]

        # --- Generate Word Boundaries ---
        # word_boundaries[i] = True if token i starts a new "word" (space-separated token in prefix notation)
        all_word_boundaries = []
        for i in range(len(model_inputs["input_ids"])): # Iterate through each example in the batch
            current_input_ids = model_inputs["input_ids"][i]
            tokens = tokenizer.convert_ids_to_tokens(current_input_ids)
            word_boundaries = [False] * len(tokens)

            for j, token in enumerate(tokens):
                # SentencePiece uses ' ' (U+2581) to mark start-of-word pieces.
                # This is a reliable way to detect word starts for many tokenizers.
                if token.startswith(" "):
                    word_boundaries[j] = True
                # Handle the very first token if it's not a special token or space-prefixed
                # Special tokens depend on the tokenizer (e.g., <pad>, </s>, [CLS])
                elif j == 0 and token not in tokenizer.all_special_tokens:
                    word_boundaries[j] = True
                # Sometimes the first token *is* a space marker, handled above.
                # If the first token isn't special and doesn't start with space, it's likely the start.

            # Ensure the list isn't empty and handle edge case of only special tokens
            if not any(word_boundaries) and len(word_boundaries) > 0:
                # If no token started with a space (e.g., single word, or odd tokenization),
                # mark the first non-special token as the start.
                first_real_token_idx = -1
                for k, token in enumerate(tokens):
                    if token not in tokenizer.all_special_tokens:
                        first_real_token_idx = k
                        break
                if first_real_token_idx != -1:
                    word_boundaries[first_real_token_idx] = True

            all_word_boundaries.append(word_boundaries)

        model_inputs["word_boundaries"] = all_word_boundaries

        return model_inputs


    # Apply tokenization if dataset loaded successfully
    tokenized_datasets = None # Initialize
    if dataset and len(dataset['train']) > 0: # Ensure train set is not empty
        # Optional: Filter out examples where tokenization might result in empty sequences after special tokens
        # This is less common with prefix notation but good practice
        original_len_train = len(dataset['train'])
        original_len_test = len(dataset['test'])

        # Define filter function (adapt if necessary based on tokenizer)
        # T5 usually adds EOS, so length > 1 is safe. Check prefix adds EOS too.
        def filter_short_sequences(example):
            # Check length *after* adding potential prefix and special tokens
            task_prefix = "translate prefix equation to prefix solution: "
            input_len = len(tokenizer(task_prefix + str(example['equ_prefix']))['input_ids'])
            # Target length check depends on target tokenizer behavior
            with tokenizer.as_target_tokenizer():
                target_len = len(tokenizer(str(example['sol_prefix']))['input_ids'])
            return input_len > 1 and target_len > 1 # Keep examples with at least one token + EOS

        dataset = dataset.filter(filter_short_sequences)

        filtered_len_train = len(dataset['train'])
        filtered_len_test = len(dataset['test'])

        if original_len_train > filtered_len_train:
            print(f"Filtered out {original_len_train - filtered_len_train} examples from train set due to short sequence length after tokenization.")
        if original_len_test > filtered_len_test:
            print(f"Filtered out {original_len_test - filtered_len_test} examples from test set due to short sequence length after tokenization.")


        if len(dataset['train']) == 0:
            print("Warning: After filtering, the training dataset is empty. Cannot proceed with tokenization.")
            # tokenized_datasets remains None
        else:
            print("\nStarting tokenization...")
            # Determine columns to remove: all original columns EXCEPT those needed later
            # We need 'sol_str' later for evaluation if it exists.
            columns_to_remove = list(dataset['train'].column_names)
            if 'sol_str' in columns_to_remove:
                # Don't remove sol_str if we want to use it in evaluation
                columns_to_remove.remove('sol_str')
                print("Keeping 'sol_str' column for potential SymPy evaluation.")


            tokenized_datasets = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=columns_to_remove # Remove original prefix cols, keep sol_str if present
            )
            print("Tokenization complete.")
            print("\nTokenized dataset features:")
            print(tokenized_datasets['train'].features) # Show features after tokenization

            print("\nTokenized dataset sample (first train example):")
            # Ensure accessing index 0 is safe
            if len(tokenized_datasets["train"]) > 0:
                print(tokenized_datasets["train"][0])
            else:
                print("Train split is empty after tokenization/filtering.")

            # Verify word boundaries calculation for a sample
            if len(tokenized_datasets["train"]) > 0:
                sample_idx = 0
                print("\n--- Word Boundary Check (Sample 0) ---")
                input_ids = tokenized_datasets["train"][sample_idx]['input_ids']
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                boundaries = tokenized_datasets["train"][sample_idx]['word_boundaries']
                print("Tokens:", tokens)
                print("Boundaries:", boundaries)
                print("---------------------------------------")
    else:
        print("\nDataset is empty or not loaded properly. Skipping tokenization.")
        # tokenized_datasets remains None

    # Data Collator
    # Initialize even if tokenized_datasets is None, subsequent checks handle it
    # Use padding=True to pad batches to the length of the longest sequence in the batch
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    print("\nData Collator initialized.")

    # You can now use 'tokenized_datasets' and 'data_collator' in the Trainer setup,
    # after checking that 'tokenized_datasets' is not None.
    if tokenized_datasets:
        print(f"\nSuccessfully created tokenized_datasets with {len(tokenized_datasets['train'])} training examples.")
    else:
        print("\nFailed to create tokenized_datasets.")
        
    return tokenized_datasets, data_collator