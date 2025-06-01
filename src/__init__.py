"""
Medibot 3.0 - Advanced Medical AI Chatbot
"""

__version__ = "3.0.0"
__author__ = "Zeeshan"
__email__ = "your.email@example.com"

from .medibot import MedibotInference
from .training import MedibotTrainer
from .utils import load_medical_cases, format_medical_prompt

__all__ = [
    "MedibotInference",
    "MedibotTrainer", 
    "load_medical_cases",
    "format_medical_prompt"
]
