# Medibot 3.0 🏥🤖

A sophisticated medical AI chatbot built using fine-tuned DeepSeek-R1-Distill-Llama-8B model with advanced clinical reasoning capabilities.

## 🌟 Features

- **Advanced Medical Reasoning**: Uses chain-of-thought prompting with `<think>` tags for transparent clinical reasoning
- **Specialized Training**: Fine-tuned on medical reasoning dataset with 500 clinical cases
- **Memory Efficient**: Implements LoRA (Low-Rank Adaptation) with 4-bit quantization
- **Clinical Focus**: Specialized in diagnostics, treatment planning, and medical consultation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Hugging Face account with API token
- Weights & Biases account (optional, for training monitoring)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Zeeshan2912/Medibot_3.0.git
cd Medibot_3.0
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_API_TOKEN="your_wandb_token"  # Optional
```

4. Validate your setup:
```bash
python validate_setup.py
```

### Quick Demo

For a quick demonstration:
```bash
# Run the interactive example
python example_usage.py
```

For training:
```bash
# Train with default settings
python train_model.py --hf_token YOUR_HF_TOKEN

# Or follow the Jupyter notebook for detailed training
jupyter notebook AI_Doctor_3.ipynb
```

### Usage

#### Training the Model

Open and run the Jupyter notebook:
```bash
jupyter notebook AI_Doctor_3.ipynb
```

Follow the cells sequentially to:
1. Set up the environment
2. Load the DeepSeek-R1-Distill-Llama-8B model
3. Fine-tune on medical reasoning dataset
4. Test the trained model

#### Using the Trained Model

**Option 1: Interactive Example Script**
```bash
python example_usage.py
```

**Option 2: Python API**
```python
from src.medibot import MedibotInference

# Initialize the chatbot
medibot = MedibotInference(model_path="path/to/trained/model")

# Ask a medical question
question = "A patient presents with chest pain and shortness of breath..."
response = medibot.generate_response(question)
print(response)
```

**Option 3: Command Line Training**
```bash
python train_model.py --hf_token YOUR_TOKEN --max_steps 100
```

## 🏗️ Project Structure

```
Medibot_3.0/
├── AI_Doctor_3.ipynb          # Main training notebook
├── example_usage.py           # Interactive example script
├── train_model.py             # Command-line training script
├── validate_setup.py          # Setup validation script
├── src/
│   ├── __init__.py
│   ├── medibot.py             # Core inference class
│   ├── training.py            # Training utilities
│   └── utils.py               # Helper functions
├── data/
│   └── sample_cases.json      # Sample medical cases
├── models/                    # Trained model storage
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🔬 Model Details

- **Base Model**: DeepSeek-R1-Distill-Llama-8B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: FreedomIntelligence/medical-o1-reasoning-SFT
- **Training Samples**: 500 medical reasoning cases
- **Max Sequence Length**: 2048 tokens

## 📊 Performance

The model demonstrates strong performance in:
- Clinical reasoning and diagnosis
- Treatment planning
- Medical case analysis
- Transparent thought processes

## ⚠️ Important Disclaimer

**This AI model is for educational and research purposes only. It should NOT be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DeepSeek team for the base model
- Unsloth for efficient fine-tuning framework
- FreedomIntelligence for the medical reasoning dataset
- Hugging Face for the transformers library

## 📞 Contact

For questions or support, please open an issue on GitHub.
