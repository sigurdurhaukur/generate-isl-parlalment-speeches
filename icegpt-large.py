from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    AdamW,
)
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from datasets import load_dataset
from utils import print_gpu_utilization, print_summary, get_all_paths_in_dir
from tokenizer import train_new_tokenizer


"""
To run this script you need to run the following command:

deepspeed --num_gpus=1 ./icegpt-large.py \
--deepspeed ./ds_config.json \
--model_name_or_path icegpt-xl --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name igc

"""

scale = 1
output_dir = "icegpt-xl"

# Suppress warnings
print("\n" * 3)

# Print GPU utilization
print_gpu_utilization()

# Collect paths
print("Collecting all paths...")
all_paths = get_all_paths_in_dir("./processed_data", max_paths=1000)

# Load or train tokenizer
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# base_vocab = list(bytes_to_unicode().values())
# new_tokenizer = train_new_tokenizer(tokenizer, all_paths, base_vocab, path="./ice-tokenizer-large")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./ice-tokenizer-large")
print("Done.", tokenizer)

# Split train and test paths
train_paths = all_paths[: int(len(all_paths) * 0.8)]
test_paths = all_paths[int(len(all_paths) * 0.8) :]

# Load dataset
dataset = load_dataset("text", data_files={"train": train_paths, "test": test_paths})

# GPU utilization
print_gpu_utilization()

context_length = 512

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None
)


def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=context_length,
        padding="max_length",
    )
    return outputs


# Tokenize dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    num_proc=4,
    remove_columns=["text"],
)


# Model configuration 140M parameters
# configuration = GPT2Config(
#     vocab_size=len(tokenizer),
#     n_ctx=context_length,
#     bos_token_id=tokenizer.bos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
#     n_embd=int(768),
#     n_layer=int(12),
#     n_head=int(12),
# )

# 1.23B parameters
configuration = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=context_length,
    n_embd=1750,
    n_layer=32,
    n_head=25,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Initialize model
model = GPT2LMHeadModel(configuration)

# Model size
model_size = sum(t.numel() for t in model.parameters())
print(f"ICE GPT-2 size: {model_size/1000**2:.1f}M parameters")

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    deepspeed="./ds_config.json",
)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    # optimizer=optimizer,
)

# Start training
trainer.train()

# Save trained model
model.save_pretrained(output_dir)
