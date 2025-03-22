# AI Cancer Detection - Jupyter Notebook

This notebook demonstrates the use of a neural network for cancer detection using TensorFlow and Keras. The model is trained on a dataset to predict whether a tumor is malignant (1) or benign (0).

---

## Code Walkthrough

### 1. Importing Libraries
```python
import pandas as pd
```

### 2. Loading the Dataset
The dataset is loaded from a CSV file named `cancer.csv`.
```python
dataset = pd.read_csv('cancer.csv')
```

### 3. Preparing the Data
The dataset is split into features (`x`) and labels (`y`).
```python
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]
```

### 4. Splitting the Data
The dataset is split into training and testing sets using `train_test_split` from `scikit-learn`.
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```

### 5. Building the Neural Network Model
The model is built using TensorFlow's Keras API. It consists of:
- **Input Layer**: The number of input features is determined by the shape of the training data.
- **Hidden Layers**: Two dense layers with 256 neurons each, using the sigmoid activation function.
- **Output Layer**: A single neuron with a sigmoid activation function for binary classification.

```python
import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
```

### 6. Compiling the Model
The model is compiled using the Adam optimizer and binary cross-entropy loss function. Accuracy is used as the evaluation metric.
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### 7. Training the Model
The model is trained for 1000 epochs on the training dataset.
```python
model.fit(x_train, y_train, epochs=1000)
```

---

## Training Output
The training process logs the accuracy and loss for each epoch. Here are some example outputs:

```
Epoch 1/1000
15/15 [==============================] - 0s 8ms/step - accuracy: 0.9770 - loss: 0.0585
Epoch 2/1000
15/15 [==============================] - 0s 8ms/step - accuracy: 0.9858 - loss: 0.0451
...
Epoch 1000/1000
15/15 [==============================] - 0s 5ms/step - accuracy: 0.9721 - loss: 0.0720
```

---

## Results
After training, the model's performance can be evaluated on the test dataset. The notebook includes code to evaluate the model's accuracy and loss on the test set.

---

## Dependencies
To run this notebook, you need the following Python libraries:
- `pandas`
- `tensorflow`
- `scikit-learn`

You can install the required libraries using pip:
```bash
pip install pandas tensorflow scikit-learn
```

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-cancer-detection.git
   cd ai-cancer-detection
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook AI_cancer.ipynb
   ```

3. Run the cells in the notebook to load the dataset, build the model, train it, and evaluate its performance.

---

## License
This project is open-source. Feel free to modify and use it for educational purposes.
