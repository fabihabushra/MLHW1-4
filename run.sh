#!/bin/bash

# Run Decision Tree on Car Dataset
python main.py --algorithm decisiontree --dataset car --max_depth 6

# Run Decision Tree on Bank Dataset with Unknown Values Handling
python main.py --algorithm decisiontree --dataset bank --max_depth 16 --handle_unknown y

# Run AdaBoost on Bank Dataset
python main.py --algorithm adaboost --dataset bank --iterations 500

# Run Bagging on Bank Dataset
python main.py --algorithm bagging --dataset bank --iterations 500

# Run Random Forest on Bank Dataset
python main.py --algorithm randomforest --dataset bank --iterations 500 --max_features 4,6,8

# Run Batch Gradient Descent on Concrete Dataset
python main.py --algorithm batch_gradient_descent --dataset concrete --initial_learning_rate 0.001 --decay_factor 0.8 --decay_interval 500

# Run Stochastic Gradient Descent on Concrete Dataset
python main.py --algorithm stochastic_gradient_descent --dataset concrete --initial_learning_rate 0.01 --decay_factor 0.7 --decay_interval 300

# Run Analytical Solution on Concrete Dataset
python main.py --algorithm analytical_solution --dataset concrete

# Run Bias-Variance Decomposition for Bagging on Bank Dataset
python main.py --algorithm bagging_bias_variance --dataset bank --num_runs 100 --sample_size 1000 --num_trees 500

# Run Bias-Variance Decomposition for Random Forest on Bank Dataset
python main.py --algorithm random_forest_bias_variance --dataset bank --num_runs 100 --sample_size 1000 --num_trees 500 --bias_variance_max_features 4

# Run Standard Perceptron on Banknote Dataset
python main.py --algorithm standard_perceptron --dataset bank-note --epochs 10

# Run Voted Perceptron on Banknote Dataset
python main.py --algorithm voted_perceptron --dataset bank-note --epochs 10

# Run Averaged Perceptron on Banknote Dataset
python main.py --algorithm average_perceptron --dataset bank-note --epochs 10

# Run Kernel Perceptron on Banknote Dataset
python main.py --algorithm kernel_perceptron --dataset bank-note

# Run SVM in Primal Domain on Banknote Dataset
python main.py --algorithm svm_primal --dataset bank-note --epochs 100

# Run SVM in Dual Domain on Banknote Dataset
python main.py --algorithm svm_dual --dataset bank-note

# Run Nonlinear SVM with Gaussian Kernel on Banknote Dataset
python main.py --algorithm nonlinear_svm --dataset bank-note

# Run Feedforward Neural Network on Banknote Dataset
python main.py --algorithm neuralnet --dataset bank-note --width 50 --epochs 20 --gamma0 0.1 --d 100.0

# Run Logistic Regression MAP Estimation on Banknote Dataset
python main.py --algorithm logistic_regression_map --dataset bank-note --gamma0 0.1 --d 100.0 --epochs 100

# Run Logistic Regression ML Estimation on Banknote Dataset
python main.py --algorithm logistic_regression_ml --dataset bank-note --gamma0 0.1 --d 10.0 --epochs 100
