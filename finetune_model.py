import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer # SFTTrainer is often used for instruction fine-tuning

# 1. Configuration
model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit" # Unsloth's pre-quantized model for efficiency
dataset_file = "data.jsonl" # Your JSONL data file
output_dir = "my_finetuned_chatbot_adapters" # Where LoRA adapters will be saved
lora_r = 16  # LoRA rank (alpha/rank ratio often 2, so lora_alpha = 32)
lora_alpha = 32
lora_dropout = 0.05
# You might need to adjust max_seq_length based on your VRAM and data
# Unsloth can often handle longer sequences than standard HF
max_seq_length = 4096 # Or 1024, 2048 depending on your data and VRAM
# For Unsloth, it can often auto-detect, but good to be aware of
# Unsloth's FastLanguageModel.for_training takes care of much of the LoRA setup

# 2. Load Model and Tokenizer with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None, # None will use torch.bfloat16 if available, else float16
    load_in_4bit = True, # This enables QLoRA
)

# Add LoRA adapters - Unsloth makes this easy
# `r` (rank), `target_modules` (layers to apply LoRA), `lora_alpha`, `lora_dropout`
# Unsloth has good defaults for target_modules for many models
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_r,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], # Common for Mistral/Llama
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    bias = "none",  # Optimal setting for Llama/Mistral
    use_gradient_checkpointing = True, # Saves memory
    random_state = 42,
    #max_seq_length = max_seq_length, # Unsloth handles this
)

# 3. Load and Prepare Dataset
# The SFTTrainer expects a column with the conversational text, often 'text' or specific to the dataset structure.
# For JSONL with "messages", you'll need a formatting function.
# Unsloth provides utility for chat templates, or you can define your own.

# Example of how to format the "messages" structure into a single string if your trainer needs it.
# Some trainers/Unsloth versions might handle the "messages" format directly if dataset is loaded appropriately.
# Check Unsloth's examples for the most current way to format chat data.

# For SFTTrainer, it's often easiest if your dataset has a 'text' column where each entry is the
# fully formatted conversation string, including special tokens for roles.
# tokenizer.apply_chat_template can do this.

def formatting_prompts_func(examples):
    convos = examples["messages"] # Assumes your dataset loads 'messages' column
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

dataset = load_dataset("json", data_files=dataset_file, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True,)
print(f"Dataset loaded and formatted. First example:\n{dataset[0]['text']}")


# 4. Training Arguments
# You can use HuggingFace TrainingArguments or Unsloth's optimized version
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,  # Adjust based on VRAM, 1 or 2 is common for 16GB QLoRA
    gradient_accumulation_steps=4,  # Effective batch size = batch_size * accumulation_steps
    warmup_steps=10,
    # max_steps=100, # Or num_train_epochs. Set one. For testing, a few steps is fine.
    num_train_epochs=1, # Start with 1-3 epochs, more can lead to overfitting
    learning_rate=2e-4, # Common for LoRA
    fp16=not torch.cuda.is_bf16_supported(), # Use bfloat16 if available (Ampere+ GPUs), else fp16
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit", # Unsloth recommends this for memory saving
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    save_strategy="epoch", # Or "steps" with save_steps
    # report_to="tensorboard" # Optional for tracking
)

# 5. Initialize Trainer (SFTTrainer for instruction/chat fine-tuning)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # The column with your formatted conversations
    max_seq_length=max_seq_length,
    args=training_args,
    # packing = False, # For chatml, Unsloth might handle this. Packing can speed up training.
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# 6. Start Fine-tuning
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning finished!")

# 7. Save the LoRA Adapters
print(f"Saving LoRA adapters to {output_dir}")
model.save_pretrained(output_dir) # Saves LoRA adapters

# You can also save the full model if you merge LoRA weights,
# but saving adapters is more common and space-efficient for LoRA.
# If you want to save a GGUF or other formats, Unsloth has export utilities.
# model.save_pretrained_gguf(output_dir, tokenizer, quantization_method = "q4_k_m") # Example for GGUF

print("Script finished successfully!")