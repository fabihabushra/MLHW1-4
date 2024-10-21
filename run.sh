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
python main.py --algorithm stochastic_gradient_descent --dataset concrete --initial_learning_rate 0.01 --decay_factor 0.8 --decay_interval 500

# Run Analytical Solution on Concrete Dataset
python main.py --algorithm analytical_solution --dataset concrete

# Run Bias-Variance Decomposition for Bagging on Bank Dataset
python main.py --algorithm bagging_bias_variance --dataset bank --num_runs 100 --sample_size 1000 --num_trees 500

# Run Bias-Variance Decomposition for Random Forest on Bank Dataset
python main.py --algorithm random_forest_bias_variance --dataset bank --num_runs 100 --sample_size 1000 --num_trees 500 --bias_variance_max_features 4
