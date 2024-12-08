# Titanic Kaggle Challenge

This project aims to predict passenger survival on the Titanic using machine learning. The dataset is part of the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition on Kaggle.

## Project Overview

In this project, we use the Titanic dataset to predict whether a passenger survived or not based on various features like age, sex, class, and the number of siblings/spouses aboard. The dataset consists of both categorical and numerical data, which we preprocess before training a machine learning model.

### Key Steps:
- **Data Preprocessing**: Clean and prepare the data by handling missing values, encoding categorical variables, and splitting data for training and testing.
- **Exploratory Data Analysis (EDA)**: Visualize and explore the dataset to gain insights into the relationships between the features and survival rate.
- **Model Training**: Train a Random Forest Classifier model to predict survival based on the features in the dataset.
- **Evaluation**: Evaluate the model's performance using accuracy score and confusion matrix.

## Requirements

To run this project, you need the following Python libraries:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
