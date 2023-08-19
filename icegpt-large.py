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
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from utils import print_gpu_utilization, print_summary, get_all_paths_in_dir
from tokenizer import train_new_tokenizer

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

scale = 1

# Model configuration
configuration = GPT2Config(
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_embd=int(768 * scale),
    n_layer=int(12 * scale),
    n_head=int(12 * scale),
)

# Initialize model
model = GPT2LMHeadModel(configuration)

# Model size
model_size = sum(t.numel() for t in model.parameters())
print(f"ICE GPT-2 size: {model_size/1000**2:.1f}M parameters")

# Training arguments
training_args = TrainingArguments(
    output_dir="icegpt-large",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
)

# AdamW optimizer
adam_bnb_optim = AdamW(
    model.parameters(),
    lr=training_args.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=training_args.weight_decay,
)

# DataLoader
dataloader = DataLoader(
    tokenized_dataset["train"], batch_size=training_args.per_device_train_batch_size
)

# Optional gradient checkpointing
if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save trained model
model.save_pretrained("ice-gpt2")
