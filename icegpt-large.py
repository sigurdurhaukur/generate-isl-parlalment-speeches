from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second{result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


print_gpu_utilization()


from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

# Base tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
base_vocab = list(bytes_to_unicode().values())


# Training and saving
new_tokenizer = tokenizer.train_new_from_iterator(
    all_paths, vocab_size=30000, initial_alphabet=base_vocab, show_progress=True
)

new_tokenizer.add_special_tokens(
    {
        "eos_token": "<|endoftext|>",
        "pad_token": "<pad>",
    }
)

new_tokenizer.save_pretrained("ice-tokenizer-large")


# load tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./ice-tokenizer-large")

# tokenizer.pad_token = "<pad>"


# tokenizer('Hvað er að frétta 😁 ?')

# get special tokens
tokenizer.cls_token, tokenizer.sep_token, tokenizer.unk_token, tokenizer.pad_token, tokenizer.mask_token


import os

all_paths = []

print("Collecting all paths...")
for root, dirs, files in os.walk("./processed_data"):
    for file in files:
        if file.endswith(".txt"):
            all_paths.append(os.path.join(root, file))


from datasets import load_dataset

portion = 1
all_paths = all_paths[: int(len(all_paths) * portion)]

train_paths = all_paths[: int(len(all_paths) * 0.8)]
test_paths = all_paths[int(len(all_paths) * 0.8) :]

# train and test split
dataset = load_dataset("text", data_files={"train": train_paths, "test": test_paths})
dataset


print_gpu_utilization()


tokenizer(
    "I", truncation=False, padding="max_length", max_length=5, return_tensors="pt"
)


import torch
from transformers import DataCollatorForLanguageModeling

context_length = 512


def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=context_length,
        padding="max_length",  # padds to the right by default
        # return_overflowing_tokens=True,
        # return_length=True,
    )
    return outputs


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None
)


tokanized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    num_proc=4,
    remove_columns=["text"],
)


tokanized_dataset


tokanized_dataset["train"][2]["input_ids"]  # 1024


from transformers import GPT2Config, GPT2LMHeadModel

scale = 1

# Initializing a GPT2 configuration
configuration = GPT2Config(
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_embd=int(768 * scale),
    n_layer=int(12 * scale),
    n_head=int(12 * scale),
)

# Initializing a model from the configuration
model = GPT2LMHeadModel(configuration)

# Accessing the model configuration
configuration = model.config


model.config


model_size = sum(t.numel() for t in model.parameters())
print(f"ICE GPT-2 size: {model_size/1000**2:.1f}M parameters")


model.to("cuda")

print_gpu_utilization()


from transformers import Trainer, TrainingArguments
from transformers import AdamW


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

adam_bnb_optim = AdamW(
    model.parameters(),
    lr=training_args.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=training_args.weight_decay,
)


from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
import torch

dataloader = DataLoader(
    tokanized_dataset["train"], batch_size=training_args.per_device_train_batch_size
)

if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()


accelerator = Accelerator()


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokanized_dataset["train"],
    eval_dataset=tokanized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()


model.save_pretrained("ice-gpt2")


from transformers import TextGenerationPipeline

# Load the trained model
model_path = "ice-gpt2"
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# Create a text generation pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

# Generate text
generated_text = pipeline(
    "hæ, hæ",
    max_length=1000,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
)

# Print the generated text
print(generated_text[0]["generated_text"])


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame(trainer.state.log_history)
df.to_csv("./ice-gpt2/log_history.csv")

training_loss_data = []
eval_loss_data = []
for i in range(len(df)):
    row = df.iloc[i]
    epoch, train_loss, eval_loss = row["epoch"], row["loss"], row["eval_loss"]

    if not np.isnan(eval_loss):
        eval_loss_data.append((epoch, eval_loss))

    if not np.isnan(train_loss):
        training_loss_data.append((epoch, train_loss))

# Extract x and y values for training and evaluation losses
training_epochs, training_losses = zip(*training_loss_data)
eval_epochs, eval_losses = zip(*eval_loss_data)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the training loss
ax.plot(training_epochs, training_losses, label="Training Loss")

# Plot the evaluation loss
ax.plot(eval_epochs, eval_losses, label="Evaluation Loss")

# Set labels and title
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Training and Evaluation Loss")

# Add a legend
ax.legend()

# Display the plot
plt.show()
