# Iris Flower Classification

This is a simple machine learning project that classifies Iris flowers into three species trying multiple classification algorithms. It's my first time using python to explore data and apply some machine learning. This project starts from data exploration and goes until model evaluation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## 🎯 Overview

This project tries six different machine learning algorithms to classify Iris flowers based on their sepal and petal measurements. The project includes comprehensive data visualization, exploratory data analysis, and model evaluation.

## ✨ What will you find

- **Data Exploration**: Statistical summaries and visualizations including violin plots, pair plots, and correlation heatmaps
- **Multiple Algorithms**: Comparison of 6 different classification algorithms:
  - Linear Discriminant Analysis (LDA)
  - K-Nearest Neighbors (KNN)
  - Decision Tree (DT)
  - Naive Bayes (NB)
  - Support Vector Classifier (SVC)
  - Random Forest (RF)
- **Model Evaluation**: Checking the performance of the models
- **Visualization**: Confusion matrix heatmap and classification report

## 📊 Dataset

The project uses the classic [Iris dataset](iris.csv), which contains:

- **150 samples** (50 per class)
- **4 features**: Sepal Length, Sepal Width, Petal Length, Petal Width
- **3 classes**: Iris-setosa, Iris-versicolor, Iris-virginica

## 🛠 Technologies Used

- **Python 3**
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Basic plotting
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning algorithms and evaluation metrics

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/iris-classifier.git
cd iris-classifier
```

2. Install the required dependencies:

Run:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

Or simply install from the notebook running the first cell.

## 🚀 Usage

1. Open the [Jupyter notebook](IrisClassifier.ipynb)
2. Run the cells sequentially to:
   - Load and explore the dataset
   - Visualize data distributions and correlations
   - Train and compare multiple classification models
   - Evaluate the best model on test data
   - Generate performance metrics and visualizations

The notebook is self-contained and can be run from top to bottom.

## 📈 Results

### Model Comparison

| Algorithm | Mean Accuracy | Std Deviation |
|-----------|--------------|---------------|
| **SVC** | **0.9833** | 0.0333 |
| LDA | 0.9750 | 0.0382 |
| KNN | 0.9583 | 0.0417 |
| DT | 0.9583 | 0.0417 |
| NB | 0.9500 | 0.0553 |
| RF | 0.9417 | 0.0534 |

### Best Model Performance (SVC)

- **Test Accuracy**: 96.67%
- **Precision**: 0.97 (weighted avg)
- **Recall**: 0.97 (weighted avg)
- **F1-Score**: 0.97 (weighted avg)

The Support Vector Classifier (SVC) achieved the highest accuracy and was selected as the final model.
