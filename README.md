# ITEC-EFADS (Enhanced Fault Anomaly Detection System)

A machine learning project for battery state-of-charge (SoC) estimation using PyTorch and TensorFlow Lite.

## Project Structure

```
├── data/                   # Dataset files
│   ├── train.csv          # Training data
│   └── valid.csv          # Validation data
│   └── *_TEST_*.csv       # Test datasets at different temperatures
├── src/                   # Source code
│   ├── cpp/               # C++ inference implementation
│   ├── inference/         # TensorFlow Lite model deployment
│   ├── models/           # Model architectures
│   ├── datasets/         # Dataset handling code
│   ├── utils.py          # Utility functions
│   └── main.ipynb        # Main training notebook
├── requirements.txt       # Python dependencies
└── Dockerfile            # Container definition
```

## Overview

This project implements a battery state-of-charge estimation system using deep learning. It includes:

- Training pipeline using PyTorch Lightning
- Model deployment for embedded systems using TensorFlow Lite
- C++ inference implementation
- Docker support for reproducible environments

## Key Features

- State-of-charge estimation with RMSE < 1.5%
- Support for different temperature conditions (-10°C to 25°C)
- Embedded deployment support for Arduino
- Real-time inference capabilities

## Getting Started

### Prerequisites

- Python 3.12+
- PyTorch 2.5+
- CUDA-capable GPU (optional)

### Installation

1. Clone the repository:
```sh
git clone https://github.com/yourusername/ITEC-EFADS.git
cd ITEC-EFADS
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

3. Using Docker (alternative):
```sh
docker build -t itec-efads .
docker run -it --gpus all itec-efads
```

### Training

Open and run `src/main.ipynb` to train the model. The notebook includes:
- Data loading and preprocessing
- Model training with early stopping
- Performance evaluation
- Model export for deployment

### Deployment

For Arduino deployment, follow instructions in `src/inference/run_pt_model/README.md`.

## Performance

Test results across different temperatures:
- -10°C: RMSE 1.081%, MAE 0.773%
- 0°C: RMSE 1.226%, MAE 0.788%
- 10°C: RMSE 1.472%, MAE 0.947%
- 25°C: RMSE 1.437%, MAE 0.947%

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.