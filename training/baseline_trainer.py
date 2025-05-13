from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch

from dataset import load_dataset

tokenized_datasets, data_collator = load_dataset()

# create custom tiny T5 arch model:

from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

# first t5 config:
t5_config = T5Config(
    vocab_size=32128,
    d_model=128,
    d_ff=512,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    
)

model = T5ForConditionalGeneration(t5_config)

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results_standard",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8, # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,         # Adjust as needed
    predict_with_generate=True,
    fp16=torch.cuda.is_available(), # Enable mixed precision if CUDA is available
    logging_steps=10,
    # group_by_length=True, # Can speed up training
)

trainer_standard = Seq2SeqTrainer(
    model=model, # Use a fresh instance of the model for standard training
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer_standard.train()
print("\nStandard training finished.")
