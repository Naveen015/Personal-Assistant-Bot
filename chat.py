import torch
from unsloth import FastLanguageModel

# 1. Configuration
base_model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit" # Same base model used for fine-tuning
adapters_path = "my_finetuned_chatbot_adapters" # Path to your saved LoRA adapters
max_seq_length = 2048 # Should match or be compatible with training

# 2. Load Base Model and then apply LoRA adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = adapters_path, # Load the LoRA adapters directly
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Optimizes for inference

# 3. Chat Loop
# Define the system prompt you used (or want to use) for inference.
# This should ideally match or be compatible with the system prompt in your training data.
system_prompt_content = "You are MyAssistant, a helpful AI assistant that answers questions about [Your Name] and talks like them. You are witty, knowledgeable about [Your Interests/Work], and sometimes use [Your Specific Phrases/Mannerisms]."

history = [{"role": "system", "content": system_prompt_content}]

print("Chatbot initialized. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    current_conversation = history + [{"role": "user", "content": user_input}]

    # Prepare input for the model
    inputs = tokenizer.apply_chat_template(current_conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    # Generate response
    # Unsloth uses `generate_prompt` or you can use the standard `generate`
    outputs = model.generate(input_ids=inputs, max_new_tokens=256, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    response_ids = outputs[0][inputs.shape[-1]:] # Get only the new tokens
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    print(f"MyAssistant: {response}")
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})

    # Optional: trim history to save context window space if conversation gets too long
    # if len(history) > 10: # Keep last 5 exchanges + system prompt
    #     history = [history[0]] + history[-10:]