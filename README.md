# Fish-Speech-Simplify

This is a refactored and simplified version of the original fish-speech project. (non-author)

## ✨ Features

📖 Cleaner code structure for easier understanding

🛠️ Great for learning and educational purposes

## 📦 Installation
1. Install environments
```bash
git clone https://github.com/bittyice/fish-speech-simplify.git
cd fish-speech-simplify
pip install -r requirements.txt
```

2. Put the openaudio-s1-mini model file in the checkpoints/openaudio-s1-mini dir.

It looks like: 

fish-speech-simplify/

│── checkpoints/openaudio-s1-mini

│── dataset/            # Data processing

│── index.py            # Inference entry point

│── requirements.txt    # Dependencies

└── README.md

openaudio-s1-mini: https://huggingface.co/fishaudio/openaudio-s1-mini

## 🚀 Usage
### Inference
```bash
python index.py
```

### Fine-tuning
1. Collect Data
```bash
python data_clean.py
```

2. Train
```bash
python ft.py
```


## 🤝 Acknowledgements

Original project: fish-speech
