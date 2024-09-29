# Custom Models and Training with TensorFlow

## Project Overview
This project focuses on building custom models and training procedures using TensorFlow and Keras. The notebook provides theoretical insights, practical implementations, and visualizations to enhance understanding of how to create, compile, and train custom neural network models.

## What I've Learned

- **TensorFlow Basics**: Gained insights into TensorFlow's core functionalities, including constants, variables, and tensor operations.
- **Custom Model Building**: Learned how to build custom neural network models using the Keras API.
- **Training and Evaluation**: Explored various methods for compiling and training models, along with techniques for evaluating performance.

## Getting Started

### Prerequisites

To run this notebook, ensure you have the following:

- Python version ≥ 3.5
- TensorFlow version ≥ 2.4
- Required libraries:
  - NumPy
  - Matplotlib

You can install the necessary libraries using pip:

```bash
pip install numpy matplotlib tensorflow
```

### Running the Notebook

To execute the notebook, follow these steps:

1. Clone or download the repository containing the notebook.
2. Open the notebook in a Jupyter environment.
3. Run the cells sequentially to load data, build custom models, and visualize results.

## Key Concepts Covered

### 1. Setup

The notebook begins by importing necessary libraries and setting up the environment for reproducibility. Key imports include:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
```

### 2. Tensor Operations

Basic tensor operations are introduced using TensorFlow constants. For example, creating a matrix and inspecting its shape:

```python
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t.shape)  # Output: (2, 3)
```

### 3. Tensor Data Types

Understanding tensor data types is essential for model development. The following code demonstrates how to check the data type of a tensor:

```python
print(t.dtype)  # Output: <dtype: 'float32'>
```

### 4. Indexing and Slicing Tensors

The notebook shows how to slice and index tensors to access specific elements or sub-arrays. For example, selecting a specific column from the tensor:

```python
print(t[:, 1:])  # Selects all rows and columns from index 1 onward
```

### 5. Building Custom Models

The process of building custom models using Keras is introduced, focusing on creating layers, defining the model architecture, and compiling the model. This can include defining a sequential model:

```python
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
    keras.layers.Dense(1)  # Output layer
])
```

### 6. Training and Evaluation

The notebook illustrates how to compile and train the model with appropriate loss functions and optimizers:

```python
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10)
```

## Conclusion

This project serves as an educational resource for anyone interested in machine learning and deep learning, specifically focusing on custom models and training with TensorFlow. By exploring the TensorFlow framework, users gain a deeper understanding of how to build, train, and evaluate neural network models effectively.
