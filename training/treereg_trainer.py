# Make sure the regularizer code is importable
try:
    from regularizer.regularizer_main import TreeRegularizer
except ImportError:
    print(
        "Error: Could not import TreeRegularizer. Make sure regularizer_main.py is in the 'regularizer' directory."
    )
    TreeRegularizer = None  # Set to None to prevent errors later

from transformers import Seq2SeqTrainer


# --- Custom Trainer for TreeReg ---
class TreeRegTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        tree_reg_lambda=0.1,
        tree_reg_layer=2,
        tree_reg_heads=[0, 1],
        tree_reg_step_interval=20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if TreeRegularizer is None:
            raise ImportError("TreeRegularizer class not available.")

        self.tree_regularizer = TreeRegularizer(orth_bidir=True)
        self.tree_reg_lambda = tree_reg_lambda  # Weight for the TreeReg loss
        self.tree_reg_layer = tree_reg_layer  # 0-indexed layer for hidden states
        self.tree_reg_heads = tree_reg_heads  # List of head indices
        self.tree_reg_step_interval = tree_reg_step_interval
        print(
            f"TreeRegTrainer initialized: lambda={tree_reg_lambda}, layer={tree_reg_layer}, heads={tree_reg_heads}, interval={tree_reg_step_interval}"
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Overrides the compute_loss method to add TreeReg loss.
        """
        # Standard forward pass to get model outputs and loss
        # Request hidden states and attentions for TreeReg
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        standard_loss = outputs.loss

        # --- Tree Regularization Loss Calculation ---
        tree_reg_loss = torch.tensor(0.0).to(
            self.args.device
        )  # Initialize loss to zero
        accuracy = None

        # Apply TreeReg loss only at specified intervals and during training
        apply_tree_reg = (
            self.state.global_step > 0
            and self.state.global_step % self.tree_reg_step_interval == 0
            and model.training  # Only apply during training
        )

        if apply_tree_reg:
            # We need hidden states from the ENCODER for TreeReg on the input structure
            # Layer indices typically include embedding layer, so layer 'tree_reg_layer' might be index tree_reg_layer+1
            encoder_hidden_states = outputs.encoder_hidden_states

            if (
                encoder_hidden_states is not None
                and len(encoder_hidden_states) > self.tree_reg_layer + 1
            ):
                # Get hidden states from the specified encoder layer
                # Shape: (batch_size, sequence_length, hidden_size)
                target_hidden_states = encoder_hidden_states[
                    self.tree_reg_layer + 1
                ]  # +1 to account for embeddings

                # --- Extract states corresponding to specific heads ---
                # This part is tricky as `encoder_hidden_states` are layer outputs,
                # not raw attention head outputs. The paper applies it to attention outputs.
                # A common approximation is to use the layer's hidden state directly,
                # or potentially use `outputs.encoder_attentions`. Let's use `encoder_attentions`.

                encoder_attentions = (
                    outputs.encoder_attentions
                )  # Tuple of attention weights per layer
                if (
                    encoder_attentions is not None
                    and len(encoder_attentions) > self.tree_reg_layer
                ):
                    # Attention weights shape: (batch_size, num_heads, seq_len, seq_len)
                    # Value vectors (v) are often projected from hidden states.
                    # The TreeReg paper applies SCIN to the hidden states *associated* with heads.
                    # This often means using the main hidden state output of the layer,
                    # assuming the regularization influences how these states are formed via attention.
                    # Let's stick to using target_hidden_states from the layer output for now,
                    # as directly accessing head-specific *states* (not weights) isn't standard output.
                    # The paper might have used a custom model modification.

                    # Placeholder: Using the full hidden state from the target layer.
                    # If you need *head-specific* states, model modification might be required.
                    hidden_states_for_reg = (
                        target_hidden_states  # Shape: (batch_size, seq_len, hidden_dim)
                    )

                    # Retrieve word boundaries and parses from inputs
                    # Note: The collator pads inputs, affecting indices.
                    # We need the unpadded boundaries and parses corresponding to the *original* input length.
                    # This requires careful handling, potentially storing original lengths or masks.

                    # Let's assume 'word_boundaries' and 'parses' are correctly passed and handled by the collator/dataset.
                    # The default DataCollatorForSeq2Seq might drop extra columns like 'parses' and 'word_boundaries'.
                    # We need a Custom Data Collator or modify how these are handled.

                    # --- Simplified Approach (assuming inputs contain necessary keys) ---
                    if "word_boundaries" in inputs and "parses" in inputs:
                        word_boundaries = inputs[
                            "word_boundaries"
                        ]  # This needs custom collation
                        parses = inputs  # This needs custom collation

                        # Ensure data is in the correct format (lists, not tensors for structure)
                        # This part needs robust implementation based on custom collation.
                        # Example placeholder - this WILL NOT work without custom collation
                        processed_boundaries = []
                        processed_parses = []
                        if isinstance(word_boundaries, torch.Tensor):
                            # This needs to map back to original lengths before padding
                            # For now, let's assume they are lists passed correctly (requires custom collator)
                            print(
                                "Warning: word_boundaries is a tensor, requires custom collator to handle padding/conversion"
                            )
                            # Placeholder: try converting first element if list of lists assumed
                            if isinstance(word_boundaries[0], list):
                                processed_boundaries = word_boundaries
                            else:
                                processed_boundaries = [
                                    wb.cpu().tolist() for wb in word_boundaries
                                ]  # Simplistic if tensor list

                        if isinstance(parses, torch.Tensor):  # Should be list of dicts
                            print("Warning: parses is a tensor, expected list of dicts")
                            processed_parses = (
                                []
                            )  # Cannot proceed here without proper format
                        elif isinstance(parses, list):
                            processed_parses = parses
                        else:
                            processed_parses = []

                        if (
                            processed_boundaries
                            and processed_parses
                            and len(processed_boundaries) == len(processed_parses)
                        ):
                            try:
                                # Build the SCIN chart
                                # hidden_states_for_reg might need slicing if padded
                                charts = self.tree_regularizer.build_chart(
                                    hidden_states_for_reg,
                                    processed_boundaries,
                                    processed_parses,
                                )

                                # Get the TreeReg loss score
                                scores, accuracy = self.tree_regularizer.get_score(
                                    charts,
                                    processed_boundaries,
                                    processed_parses,
                                    self.args.device,
                                )

                                # Average the scores across the batch
                                if scores:  # Check if scores list is not empty
                                    tree_reg_loss = torch.mean(torch.stack(scores))
                                else:
                                    tree_reg_loss = torch.tensor(0.0).to(
                                        self.args.device
                                    )

                            except Exception as e:
                                print(
                                    f"Error during TreeReg calculation (step {self.state.global_step}): {e}"
                                )
                                # Don't let TreeReg errors stop training
                                tree_reg_loss = torch.tensor(0.0).to(self.args.device)
                        else:
                            if (
                                self.state.global_step
                                % (self.tree_reg_step_interval * 10)
                                == 0
                            ):  # Log less frequently
                                print(
                                    f"Skipping TreeReg at step {self.state.global_step}: Missing/invalid boundaries or parses in batch."
                                )

                    else:
                        if (
                            self.state.global_step % (self.tree_reg_step_interval * 10)
                            == 0
                        ):
                            print(
                                f"Skipping TreeReg at step {self.state.global_step}: 'word_boundaries' or 'parses' not in inputs."
                            )

            else:
                # Log if hidden states aren't available as expected
                if self.state.global_step % (self.tree_reg_step_interval * 10) == 0:
                    print(
                        f"Skipping TreeReg at step {self.state.global_step}: Encoder hidden states for layer {self.tree_reg_layer} not found."
                    )

        # Combine losses
        # Ensure tree_reg_loss requires grad if it's non-zero and computed
        if tree_reg_loss.requires_grad:
            total_loss = standard_loss + self.tree_reg_lambda * tree_reg_loss
        else:  # If TreeReg wasn't computed or had error, just use standard loss
            total_loss = standard_loss

        # Log losses (optional)
        if (
            self.is_world_process_zero()
            and self.state.global_step % self.args.logging_steps == 0
        ):
            log_data = {"loss": standard_loss.item(), "step": self.state.global_step}
            if apply_tree_reg and tree_reg_loss != 0:
                log_data["tree_reg_loss"] = (
                    tree_reg_loss.item()
                )  # Log raw tree reg loss
                log_data["total_loss"] = total_loss.item()
                if accuracy is not None:
                    log_data["tree_reg_acc"] = accuracy
            # self.log(log_data) # Use internal logging

        return (total_loss, outputs) if return_outputs else total_loss


# --- Custom Data Collator (Needed for TreeRegTrainer) ---
from transformers import DataCollatorForSeq2Seq
from dataclasses import dataclass


@dataclass
class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    Extends DataCollatorForSeq2Seq to keep 'word_boundaries' and 'parses'.
    Assumes these are already present in the dataset items.
    """

    def __call__(self, features, return_tensors=None):
        # Standard collation for input_ids, attention_mask, labels
        standard_batch = super().__call__(features, return_tensors)

        # Keep track of extra fields
        if "word_boundaries" in features[0]:
            # This needs careful handling of padding. The simplest is to keep them as lists.
            # Padding boundaries might require knowing the max_length and padding strategy.
            # For now, just pass them as a list of lists/tensors. TreeReg needs original length info.
            standard_batch["word_boundaries"] = [f["word_boundaries"] for f in features]

        if "parses" in features[0]:
            standard_batch["parses"] = [f["parses"] for f in features]
            
        return standard_batch


# --- Initialize TreeReg Trainer ---
# Reload model to start fresh training
# first t5 config:
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
)
import torch

t5_config = T5Config(
    vocab_size=32128,
    d_model=128,
    d_ff=512,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    decoder_start_token_id=0, # Set to pad token id
)

model_treereg = T5ForConditionalGeneration(t5_config)

tokenizer = T5Tokenizer.from_pretrained("t5-small")

from dataset import load_dataset

tokenized_datasets, data_collator = load_dataset()

from parse_tree import add_parses_to_batch

if tokenized_datasets:
    try:
        # Use map to add parses permanently (if memory allows)
        tokenized_datasets = tokenized_datasets.map(add_parses_to_batch, batched=True, fn_kwargs={"tokenizer": tokenizer})
        print("\nDataset with parses added:")
        if len(tokenized_datasets["train"]) > 0:
             print(tokenized_datasets["train"][0]['parses']) # Show parse for the first example
        else:
             print("Train split empty, cannot show parse example.")
    except Exception as e:
        print(f"Error adding parses to dataset: {e}")
        # Fallback or handle error
else:
    print("\nSkipping parse generation as tokenized_datasets is not available.")

# Create the custom collator
custom_collator = CustomDataCollatorForSeq2Seq(
    tokenizer, model=model_treereg, padding=True
)

# Define training arguments for TreeReg run
training_args_treereg = Seq2SeqTrainingArguments(
    output_dir="./results_treereg",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,  # Adjust as needed
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # Enable mixed precision if CUDA is available
    logging_steps=10,  # Log frequently to see TreeReg loss
    # group_by_length=True,
)


trainer_treereg = None
# Ensure the dataset HAS the 'parses' and 'word_boundaries' columns
if (
    "parses" in tokenized_datasets["train"].features
    and "word_boundaries" in tokenized_datasets["train"].features
):
    trainer_treereg = TreeRegTrainer(
        model=model_treereg,
        args=training_args_treereg,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=custom_collator,  # Use the custom collator
        tree_reg_lambda=0.1,  # Example weight, tune as needed
        tree_reg_layer=2,  # As specified (Layer 2 -> index 3 if embeddings=0)
        tree_reg_heads=[0, 1],  # Example heads (indices)
        tree_reg_step_interval=20,  # As specified
    )
    print("\nTreeReg Trainer initialized.")
else:
    print(
        "\nCould not initialize TreeReg Trainer: 'parses' or 'word_boundaries' missing from tokenized dataset features."
    )
    print("Ensure 'add_parses_to_batch' was successfully mapped.")

if trainer_treereg:
    trainer_treereg.train()
    print("\nTreeReg training finished.")

# save treereg model
model_treereg.save_pretrained("./results_treereg")
tokenizer.save_pretrained("./results_treereg")


# first, load validation dataset
dataset, data_collator = load_dataset("val.csv")

# now evaluate the model:
# using functions from eval.py
from eval import prefix_to_sympy, safe_parse_expr, check_sympy_equivalence

# now for each item in val dataset, generate the prefix solution from model, and compare to ground truth

n_correct = 0
n = 0
for i in range(len(dataset["equ_prefix"])):
    # get the prefix solution from model
    prefix_str = dataset["equ_prefix"][i]
    prefix_solution = model.generate(tokenizer(prefix_str, return_tensors="pt").input_ids, max_length=50)
    prefix_solution = tokenizer.decode(prefix_solution[0], skip_special_tokens=True)

    # get the ground truth
    ground_truth = dataset["sol_str"][i]

    # convert prefix to sympy
    prefix_solution_sympy = prefix_to_sympy(prefix_solution)
    ground_truth_sympy = safe_parse_expr(ground_truth)

    # check if they are equivalent
    if check_sympy_equivalence(prefix_solution_sympy, ground_truth_sympy):
        n_correct += 1
        
    n += 1
    
print(f"Accuracy: {n_correct / n}")