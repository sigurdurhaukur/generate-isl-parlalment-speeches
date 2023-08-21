from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    AdamW,
    HfArgumentParser,
)

import transformers

from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from datasets import load_dataset
from utils import (
    print_gpu_utilization,
    print_summary,
    get_all_paths_in_dir,
    ModelArguments,
    DataTrainingArguments,
)
from tokenizer import train_new_tokenizer
import os, sys
import logging
from transformers.trainer_utils import get_last_checkpoint, is_main_process

"""
To run this script you need to run the following command:

deepspeed --num_gpus=1 ./icegpt-large.py \
--deepspeed ./ds_config.json \
--model_name_or_path ./icegpt-xl \
--do_train \
--do_eval \
--dataset_name IGC \
--fp16 \
--overwrite_cache \
--evaluation_strategy="steps" \
--output_dir ./icegpt-xl \
--overwrite_output_dir \
--eval_steps 200 \
--num_train_epochs 1 \
--gradient_accumulation_steps 8 \
--per_device_train_batch_size 2

"""

logger = logging.getLogger(__name__)

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=os.path.abspath(sys.argv[1])
    )
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Detecting last checkpoint.
last_checkpoint = None
if (
    os.path.isdir(training_args.output_dir)
    and training_args.do_train
    and not training_args.overwrite_output_dir
):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(
    logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)

# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
logger.info("Training/evaluation parameters %s", training_args)

# Suppress warnings
print("\n" * 3)

# Print GPU utilization
print_gpu_utilization()

# Collect paths
print("Collecting all paths...")
# all_paths = get_all_paths_in_dir("./processed_data", max_paths=1000) # for testing
all_paths = get_all_paths_in_dir("./processed_data_journals")

# Load or train tokenizer
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# base_vocab = list(bytes_to_unicode().values())
# new_tokenizer = train_new_tokenizer(tokenizer, all_paths, base_vocab, path="./ice-tokenizer-large")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./ice-tokenizer-large")
print("Done.", tokenizer)

# Split train and test paths
print("Splitting train and test paths...")
train_paths = all_paths[: int(len(all_paths) * 0.8)]
test_paths = all_paths[int(len(all_paths) * 0.8) :]

# Load dataset
print("Loading dataset...")
dataset = load_dataset("text", data_files={"train": train_paths, "test": test_paths})

context_length = 512

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None
)


print(dataset["train"][0])


def tokenize_function(examples):
    # add eos token
    examples["text"] = [text + tokenizer.eos_token for text in examples["text"]]

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
    # num_proc=4,
    remove_columns=["text"],
)

print_gpu_utilization()


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


# 1B parameters
configuration = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=context_length,
    n_embd=1200,
    n_layer=24,
    n_head=12,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Initialize model
model = GPT2LMHeadModel(configuration)

# Model size
model_size = sum(t.numel() for t in model.parameters())
print(f"ICE GPT-2 size: {model_size/1000**2:.1f}M parameters")


# Training arguments

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

# training_args = TrainingArguments(
#     output_dir=output_dir,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     evaluation_strategy="steps",
#     eval_steps=5_000,
#     logging_steps=5_000,
#     gradient_accumulation_steps=4,
#     num_train_epochs=1,
#     weight_decay=0.1,
#     warmup_steps=1_000,
#     lr_scheduler_type="cosine",
#     learning_rate=5e-4,
#     save_steps=5_000,
#     fp16=True,
#     deepspeed="./ds_config.json",
# )

if data_args.block_size is None:
    block_size = tokenizer.model_max_length
    if block_size > 1024:
        logger.warn(
            f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            "Picking 1024 instead. You can change that default value by passing --block_size xxx."
        )
    block_size = 1024
else:
    if data_args.block_size > tokenizer.model_max_length:
        logger.warn(
            f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
    block_size = min(data_args.block_size, tokenizer.model_max_length)


if training_args.do_train:
    if "train" not in tokenized_dataset:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = tokenized_dataset["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

if training_args.do_eval:
    if "test" not in tokenized_dataset:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = tokenized_dataset["test"]
    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    # eval_dataset=tokenized_dataset["test"]
    # optimizer=optimizer,
)

# Training
if training_args.do_train:
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif model_args.model_name_or_path is not None and os.path.isdir(
        model_args.model_name_or_path
    ):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None
    # train_result = trainer.train(resume_from_checkpoint=checkpoint) # only works when continuing from checkpoint
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None
        else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# Evaluation
# if training_args.do_eval:
#     logger.info("*** Evaluate ***")

#     metrics = trainer.evaluate()

#     max_val_samples = (
#         data_args.max_val_samples
#         if data_args.max_val_samples is not None
#         else len(eval_dataset)
#     )
#     metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
#     perplexity = math.exp(metrics["eval_loss"])
#     metrics["perplexity"] = perplexity

#     trainer.log_metrics("eval", metrics)
#     trainer.save_metrics("eval", metrics)
