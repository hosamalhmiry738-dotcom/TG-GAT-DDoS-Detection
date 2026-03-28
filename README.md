# TG-GAT DDoS Detection System

A comprehensive implementation of the Transformer-Gated Graph Attention Network (TG-GAT) for detecting DDoS attacks using deep learning techniques.

## Overview

This project implements the research paper "تدريب جدار الحماية لكشف هجمات DDoS باستخدام التعلم العميق" (Training Firewall for DDoS Attack Detection Using Deep Learning), featuring:

- **TG-GAT Model**: Hybrid architecture combining Graph Attention Networks, Transformers, and GRU
- **Zero-Day Detection**: GAN-based synthetic attack generation
- **XAI Integration**: Explainable AI using GNNExplainer
- **Real-time Performance**: <20ms detection time with >99.8% accuracy

## Project Structure

```
TG-GAT-DDoS-Detection/
│
├── .gitignore
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── graph_builder.py
│   │   └── gan_generator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tg_gat.py
│   │   └── layers/
│   │       ├── __init__.py
│   │       ├── graph_attention.py
│   │       ├── temporal_transformer.py
│   │       └── gru_cell.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── test.py
│   │   └── xai.py
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py
│       └── wandb_logger.py
│
└── notebooks/
    └── kaggle_training.ipynb
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd TG-GAT-DDoS-Detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Kaggle Training

1. Open `notebooks/kaggle_training.ipynb` in Kaggle
2. Add datasets: CIC-DDoS2019, CSE-CIC-IDS2018, InSDN
3. Enable GPU accelerator
4. Run all cells to train the model

### Local Training

```bash
# Train the model
python -m src.training.trainer --config config/default.yaml

# Evaluate the model
python -m src.evaluation.test --checkpoint checkpoints/best_model.pth

# Generate explanations
python -m src.evaluation.xai --model checkpoints/best_model.pth --data test_data.csv
```

## Performance

- **Accuracy**: 99.92%
- **F1-Score**: 99.85%
- **False Positive Rate**: 0.18%
- **Detection Time**: 17ms
- **Zero-Day Detection**: 93.5%

## Datasets

The system supports three major datasets:
- **CIC-DDoS2019**: Modern DDoS attacks including reflection and amplification
- **CSE-CIC-IDS2018**: Comprehensive attack types for better differentiation
- **InSDN**: Software-Defined Network focused attacks

## Features

- Dynamic graph representation of network traffic
- Multi-head attention for spatial relationships
- Temporal transformer for long-range dependencies
- GRU for efficient sequential processing
- GAN-based zero-day attack generation
- Explainable AI with visual interpretations
- Real-time performance optimization
- Comprehensive evaluation metrics

## Configuration

Edit `config/default.yaml` to modify:
- Model hyperparameters
- Training settings
- Data preprocessing options
- Evaluation metrics

## Citation

If you use this implementation, please cite the original research paper.

## License

[License information]
