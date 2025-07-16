# FED_DKT: Federated Deep Knowledge Transfer for Continual Learning

This project implements a class incremental learning framework with deep knowledge transfer capabilities. The project is built with PyTorch and focuses on efficient continual learning in a federated setting.

## Project Structure

### Core Components

#### 1. Main Application (`main.py`)
The entry point of the project that handles:
- Argument parsing and configuration
- Training and evaluation pipeline setup
- Model initialization and training loop management
- Incremental learning task management

#### 2. Continual Learning Module (`continual/`)
The core implementation of the continual learning framework:

##### Model Architecture
- `DKT.py`: Implementation of Deep Knowledge Transfer mechanism
- `robust_models.py`: Base model architectures optimized for continual learning
- `robust_models_ImageNet.py`: ImageNet-specific model architectures
- `classifier.py`: Custom classifier implementations for incremental learning

##### Training and Evaluation
- `engine.py`: Core training and evaluation loops
- `factory.py`: Model and component factory functions
- `losses.py`: Custom loss functions including knowledge distillation losses
- `scaler.py`: Gradient scaling utilities for mixed precision training

##### Data Management
- `datasets.py`: Dataset implementations and data loading utilities
- `rehearsal.py`: Memory management for storing and replaying past examples
- `samplers.py`: Custom sampling strategies for balanced training
- `utils.py`: Utility functions for data processing and model management

### Key Features

1. **Deep Knowledge Transfer (DKT)**
   - Implements knowledge transfer between tasks
   - Supports duplex classifier architecture
   - Enables efficient learning of new classes while preserving old knowledge

2. **Memory Management**
   - Configurable memory size for storing exemplars
   - Multiple rehearsal strategies (random, closest, iCaRL, furthest)
   - Support for distributed memory across processes

3. **Training Flexibility**
   - Support for various optimization strategies
   - Customizable learning rate schedules
   - Mixed precision training support
   - Multiple data augmentation options

4. **Incremental Learning**
   - Configurable initial and incremental class numbers
   - Support for custom class ordering
   - Flexible evaluation scheduling
   - Optional finetuning after each incremental task

## Configuration Options

### Main Parameters
- `--initial-increment`: Number of classes in the base task
- `--increment`: Number of new classes per incremental task
- `--memory-size`: Size of rehearsal memory
- `--batch-size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate

### DKT Specific
- `--DKT`: Enable DKT mode
- `--duplex-clf`: Use duplex classifier
- `--kd`: Knowledge distillation weight
- `--distillation-tau`: Temperature for knowledge distillation

### Memory Management
- `--rehearsal`: Method for sample selection (random, closest, iCaRL, furthest)
- `--distributed-memory`: Use different memory per process
- `--oversample-memory`: Memory sample repetition factor

### Optimization
- `--opt`: Optimizer selection
- `--sched`: Learning rate scheduler
- `--warmup-epochs`: Number of warmup epochs
- `--weight-decay`: Weight decay factor

## Usage

1. Basic Training:
```bash
python main.py --data-path /path/to/dataset --initial-increment 50 --increment 10
```

2. Enable DKT:
```bash
python main.py --DKT --duplex-clf --kd 1.0 --data-path /path/to/dataset
```

3. Custom Memory Configuration:
```bash
python main.py --memory-size 2000 --rehearsal closest_token --data-path /path/to/dataset
```

## Dependencies
- PyTorch
- timm
- continuum
- CUDA (for GPU acceleration)

## Directory Structure
```
.
├── main.py                 # Main entry point
├── train.sh               # Training script
├── convert_memory.py      # Memory conversion utilities
├── continual/            # Core continual learning implementation
│   ├── DKT.py            # Deep Knowledge Transfer
│   ├── classifier.py     # Classifier implementations
│   ├── datasets.py       # Dataset handling
│   ├── engine.py        # Training/evaluation loops
│   ├── factory.py       # Component factories
│   ├── losses.py        # Loss functions
│   ├── rehearsal.py     # Memory management
│   ├── robust_models.py # Model architectures
│   └── utils.py         # Utility functions
└── options/             # Configuration files
``` 