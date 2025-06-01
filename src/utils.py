"""
Utility functions for Medibot
"""

import json
import os
from typing import List, Dict, Any

def load_medical_cases(file_path: str) -> List[Dict[str, Any]]:
    """
    Load medical cases from a JSON file
    
    Args:
        file_path (str): Path to the JSON file containing medical cases
        
    Returns:
        List[Dict]: List of medical cases
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        print(f"✅ Loaded {len(cases)} medical cases from {file_path}")
        return cases
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in file: {file_path}")
        return []

def save_medical_cases(cases: List[Dict[str, Any]], file_path: str):
    """
    Save medical cases to a JSON file
    
    Args:
        cases (List[Dict]): List of medical cases
        file_path (str): Path to save the JSON file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(cases)} medical cases to {file_path}")
    except Exception as e:
        print(f"❌ Error saving cases: {e}")

def format_medical_prompt(question: str, context: str = "") -> str:
    """
    Format a medical question with the standard prompt template
    
    Args:
        question (str): The medical question
        context (str): Additional context (optional)
        
    Returns:
        str: Formatted prompt
    """
    base_prompt = """
Below is a task description along with additional context provided in the input section. Your goal is to provide a well-reasoned response that effectively addresses the request.

Before crafting your answer, take a moment to carefully analyze the question. Develop a clear, step-by-step thought process to ensure your response is both logical and accurate.

### Task:
You are a medical expert specializing in clinical reasoning, diagnostics, and treatment planning. Answer the medical question below using your advanced knowledge.
"""
    
    if context:
        base_prompt += f"\n### Context:\n{context}"
    
    base_prompt += f"\n### Query:\n{question}\n\n### Answer:\n<think>"
    
    return base_prompt

def extract_thinking_process(response: str) -> tuple:
    """
    Extract the thinking process and final answer from a response
    
    Args:
        response (str): Full model response
        
    Returns:
        tuple: (thinking_process, final_answer)
    """
    if "<think>" in response and "</think>" in response:
        # Extract thinking process
        think_start = response.find("<think>") + 7
        think_end = response.find("</think>")
        thinking = response[think_start:think_end].strip()
        
        # Extract final answer
        answer = response[think_end + 8:].strip()
        
        return thinking, answer
    else:
        return "", response.strip()

def validate_medical_case(case: Dict[str, Any]) -> bool:
    """
    Validate that a medical case has required fields
    
    Args:
        case (Dict): Medical case dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ["question", "response"]
    optional_fields = ["context", "difficulty", "specialty", "tags"]
    
    # Check required fields
    for field in required_fields:
        if field not in case or not case[field]:
            print(f"❌ Missing required field: {field}")
            return False
    
    return True

def create_medical_case(question: str, response: str, **kwargs) -> Dict[str, Any]:
    """
    Create a standardized medical case dictionary
    
    Args:
        question (str): Medical question
        response (str): Expected response
        **kwargs: Additional fields (context, difficulty, specialty, tags)
        
    Returns:
        Dict: Standardized medical case
    """
    case = {
        "question": question,
        "response": response,
        "context": kwargs.get("context", ""),
        "difficulty": kwargs.get("difficulty", "medium"),
        "specialty": kwargs.get("specialty", "general"),
        "tags": kwargs.get("tags", [])
    }
    
    return case
