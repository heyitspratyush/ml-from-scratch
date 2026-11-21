# Machine Learning From Scratch

A learning-focused machine learning repository where algorithms are implemented from scratch using Python and NumPy. The goal is to understand the mathematics, implementation, optimization process, and evaluation of machine learning algorithms rather than relying entirely on ready-made ML libraries.

## Current Progress

### Supervised Learning
- **Regression**
  - Linear Regression
  - Gradient Descent
  - Mean Squared Error
  - Gradient calculation
  - Parameter updates
  - Model training and prediction

## Project Structure

```text
ML/
├── datasets/
│   └── raw/
├── experiments/
│   └── supervised_learning/
│       └── regression/
│           └── linear_regression/
├── notebooks/
│   └── supervised_learning/
│       └── regression/
│           └── linear_regression/
├── src/
│   └── ml_from_scratch/
│       └── supervised_learning/
│           └── regression/
│               └── linear_regression/
│                   ├── components/
│                   └── models/
├── tests/
│   └── supervised_learning/
│       └── regression/
│           └── linear_regression/
├── pyproject.toml
└── requirements.txt
```

## Environment Setup

Clone the repository and navigate to its root directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the project in editable mode:

```bash
python -m pip install -e .
```

## Running Tests

```bash
python -m pytest
```

## Running Linear Regression Experiment

```bash
python -m experiments.supervised_learning.regression.linear_regression.first_training
```

## Implementation Philosophy

- Implement ML algorithms from scratch.
- Understand the mathematics behind each algorithm.
- Separate reusable components from model implementations.
- Write tests for individual components and model integration.
- Conduct experiments to observe how model behavior changes.
- Gradually explore more advanced ML, deep learning, and neural network architectures.

## Learning Roadmap

- Mathematical foundations for machine learning
- Linear Regression
- Classification
- Evaluation metrics
- Feature engineering and data preprocessing
- Advanced supervised learning
- Neural networks and deep learning
- Transformers and Large Language Models

This repository evolves alongside the learning process, with implementations, experiments, tests, and documentation maintained together.