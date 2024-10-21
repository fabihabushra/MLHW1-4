## This is a machine learning library developed by Fabiha Bushra for CS5350/6350 in University of Utah.

## Overview
This project implements decision tree algorithms using the ID3 framework for both categorical and numerical datasets. The main datasets used are the Car Dataset and the Bank Marketing Dataset.

## Folder Structure
```plaintext
.
├── bank/               
│   ├── train.csv       
│   ├── test.csv        
├── car/               
│   ├── train.csv       
│   ├── test.csv
├── concrete/               
│   ├── train.csv       
│   ├── test.csv      
├── DecisionTree/
│   ├── decisiontree.py
├── EnsembleLearning/
│   ├── adaboost.py
│   ├── bagging.py
│   ├── randomforest.py
├── LinearRegression/
│   ├── batch_gradient_descent.py
│   ├── stochastic_gradient_descent.py
├── dataloader.py       
├── main.py             
├── run.sh              
```

## Running the Code
To execute the code, run the `run.sh` script located in the root directory.

## User Input

During execution, you will be prompted to:

1. **Select the dataset:**
   - Enter `1` for Car Dataset
   - Enter `2` for Bank Dataset

2. **Enter the maximum depth of the decision tree:**  
   Specify the desired maximum depth (e.g., `6` for the Car Dataset or `16` for the Bank Dataset).

3. **Indicate whether to handle "unknown" values:**  
   You will be asked if you want to replace "unknown" values with the majority of other attribute values in the training set. Enter `y` for yes or `n` for no.
