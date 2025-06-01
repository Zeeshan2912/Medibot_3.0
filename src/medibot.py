"""
Core Medibot inference class for medical consultations
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
import warnings
warnings.filterwarnings("ignore")

class MedibotInference:
    def __init__(self, model_path=None, max_length=2048, device=None):
        """
        Initialize the Medibot inference engine
        
        Args:
            model_path (str): Path to the trained model
            max_length (int): Maximum sequence length
            device (str): Device to run inference on
        """
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path:
            self.load_model(model_path)
        else:
            # Load default pre-trained model
            self.load_default_model()
    
    def load_default_model(self):
        """Load the default DeepSeek-R1-Distill-Llama-8B model"""
        model_name = "dee/DeepSeek-R1-Distill-Llama-8B"
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_length,
            dtype=None,
            load_in_4bit=True,
        )
        
        FastLanguageModel.for_inference(self.model)
        print(f"✅ Loaded default model: {model_name}")
    
    def load_model(self, model_path):
        """Load a fine-tuned model from path"""
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=self.max_length,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            print(f"✅ Loaded fine-tuned model from: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Falling back to default model...")
            self.load_default_model()
    
    def format_prompt(self, question):
        """Format the medical question with the appropriate prompt template"""
        prompt_template = """
Below is a task description along with additional context provided in the input section. Your goal is to provide a well-reasoned response that effectively addresses the request.

Before crafting your answer, take a moment to carefully analyze the question. Develop a clear, step-by-step thought process to ensure your response is both logical and accurate.

### Task:
You are a medical expert specializing in clinical reasoning, diagnostics, and treatment planning. Answer the medical question below using your advanced knowledge.

### Query:
{}

### Answer:
<think>"""
        
        return prompt_template.format(question)
    
    def generate_response(self, question, max_new_tokens=1200, temperature=0.7):
        """
        Generate a medical response for the given question
        
        Args:
            question (str): Medical question or case
            max_new_tokens (int): Maximum new tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated medical response
        """
        try:
            # Format the prompt
            formatted_prompt = self.format_prompt(question)
            
            # Tokenize input
            inputs = self.tokenizer(
                [formatted_prompt], 
                return_tensors="pt", 
                truncation=True,
                max_length=self.max_length - max_new_tokens
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part
            if "### Answer:" in full_response:
                response = full_response.split("### Answer:")[1].strip()
            else:
                response = full_response
                
            return response
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def batch_generate(self, questions, max_new_tokens=1200):
        """Generate responses for multiple questions"""
        responses = []
        for question in questions:
            response = self.generate_response(question, max_new_tokens)
            responses.append(response)
        return responses
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            "model_name": getattr(self.model, 'name_or_path', 'Unknown'),
            "device": self.device,
            "max_length": self.max_length,
            "vocab_size": len(self.tokenizer),
            "model_size": sum(p.numel() for p in self.model.parameters())
        }
