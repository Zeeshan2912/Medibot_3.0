"""
Training utilities for Medibot fine-tuning
"""

import torch
import wandb
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

class MedibotTrainer:
    def __init__(self, model_name="dee/DeepSeek-R1-Distill-Llama-8B", max_seq_length=2048):
        """
        Initialize the Medibot trainer
        
        Args:
            model_name (str): Pre-trained model name
            max_seq_length (int): Maximum sequence length
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        
    def load_model(self, hf_token=None):
        """Load the pre-trained model"""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
            token=hf_token
        )
        print(f"✅ Model loaded: {self.model_name}")
        
    def setup_lora(self, r=16, lora_alpha=16, lora_dropout=0):
        """Setup LoRA configuration"""
        self.model = FastLanguageModel.get_peft_model(
            model=self.model,
            r=r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3047,
            use_rslora=False,
            loftq_config=None
        )
        print("✅ LoRA configuration applied")
        
    def load_dataset(self, dataset_name="FreedomIntelligence/medical-o1-reasoning-SFT", 
                     split="train[:500]"):
        """Load and preprocess the training dataset"""
        dataset = load_dataset(dataset_name, "en", split=split, trust_remote_code=True)
        
        # Format training prompts
        train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

        def preprocess_function(examples):
            inputs = examples["Question"]
            cots = examples["Complex_CoT"]
            outputs = examples["Response"]
            
            texts = []
            for input_text, cot, output in zip(inputs, cots, outputs):
                text = train_prompt_style.format(input_text, cot, output) + self.tokenizer.eos_token
                texts.append(text)
            
            return {"texts": texts}
        
        processed_dataset = dataset.map(preprocess_function, batched=True)
        print(f"✅ Dataset loaded and processed: {len(processed_dataset)} samples")
        return processed_dataset
        
    def train(self, dataset, output_dir="./outputs", **training_kwargs):
        """Train the model"""
        # Default training arguments
        default_args = {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 1,
            "warmup_steps": 5,
            "max_steps": 60,
            "learning_rate": 2e-4,
            "logging_steps": 10,
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "seed": 3407,
            "output_dir": output_dir,
            "fp16": not is_bfloat16_supported(),
            "bf16": is_bfloat16_supported(),
        }
        
        # Update with user-provided arguments
        default_args.update(training_kwargs)
        
        # Clean up model for training
        if hasattr(self.model, '_unwrapped_old_generate'):
            del self.model._unwrapped_old_generate
            
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="texts",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=1,
            args=TrainingArguments(**default_args),
        )
        
        # Start training
        print("🚀 Starting training...")
        trainer_stats = trainer.train()
        print("✅ Training completed!")
        
        return trainer_stats
        
    def save_model(self, save_path):
        """Save the trained model"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"✅ Model saved to: {save_path}")
