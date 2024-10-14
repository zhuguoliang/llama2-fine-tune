import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import find_all_linear_names_nobnb, print_trainable_parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import wandb
wandb.init(mode="disabled")

output_dir="./results"
model_name ="/workspace/hongzhu/model_dir/NousResearch/Llama-2-7b-hf"

dataset_name = "/workspace/hongzhu/dataset_dir/openassistant-guanaco"
dataset = load_dataset(dataset_name, split="train")

# load model in f32 precision
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
base_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

tmodule = find_all_linear_names_nobnb(base_model)
print(tmodule)

# Change the LORA hyperparameters accordingly to fit your use case
peft_config = LoraConfig(
    r=128,
    lora_alpha=16,
    # target_modules=find_all_linear_names(base_model),
    target_modules=tmodule,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, peft_config)
print_trainable_parameters(base_model)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"An AI tool that corrects and rephrase user text grammar errors delimited by triple backticks to standard English.\n### Input: ```{example['prompt'][i]}```\n ### Output: {example['completion'][i]}"
        output_texts.append(text)
    return output_texts

# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing =True,
    max_grad_norm= 0.3,
    num_train_epochs=1, 
    learning_rate=2e-4,
    #remove bf16 true config
    # bf16=True,
    save_total_limit=3,
    logging_steps=10,
    output_dir=output_dir,
    # optim="paged_adamw_32bit",
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)

trainer = SFTTrainer(
    base_model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=2048,
    formatting_func=formatting_prompts_func,
    args=training_args
)

trainer.train() 
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)