# Machine Learning Algorithms Library

This is a machine learning library developed by Fabiha Bushra for CS5350/6350 at the University of Utah.

## Overview

This project implements various machine learning algorithms, including:

- **Decision Trees** using the ID3 algorithm
- **Ensemble Learning Methods**:
  - AdaBoost
  - Bagging
  - Random Forest
- **Linear Regression**:
  - Batch Gradient Descent
  - Stochastic Gradient Descent
  - Analytical Solution
- **Bias-Variance Decomposition** experiments for Bagging and Random Forest

The datasets used in this project are:

- **Car Evaluation Dataset** (`car`)
- **Bank Marketing Dataset** (`bank`)
- **Concrete Compressive Strength Dataset** (`concrete`)

## Folder Structure

~~~
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
│   └── decisiontree.py
├── EnsembleLearning/
│   ├── adaboost.py
│   ├── bagging.py
│   ├── randomforest.py
│   ├── bias_variance_decomposition.py
│   └── random_forest_bias_variance.py
├── LinearRegression/
│   ├── batch_gradient_descent.py
│   ├── stochastic_gradient_descent.py
│   └── analytical_solution.py
├── results/
│   └── (output plots)
├── dataloader.py       
├── main.py

~~~

## Running the Code

Run the `main.py` for different algorithms on different datasets by specifying command-line arguments.

### Usage

~~~
python main.py --algorithm <algorithm> --dataset <dataset> [additional arguments]
~~~

### Available Algorithms

- `decisiontree`
- `adaboost`
- `bagging`
- `randomforest`
- `batch_gradient_descent`
- `stochastic_gradient_descent`
- `analytical_solution`
- `bagging_bias_variance`
- `random_forest_bias_variance`

### Available Datasets

- `car`
- `bank`
- `concrete`

### Common Arguments

- `--algorithm`: The algorithm to run. Choices are listed above.
- `--dataset`: The dataset to use. Choices are listed above.

### Decision Tree Specific Arguments

- `--max_depth`: Maximum depth of the decision tree (default: no limit)
- `--handle_unknown`: For the `bank` dataset, specify whether to handle 'unknown' values by replacing them with the majority value (`y` or `n`)

### Ensemble Learning Specific Arguments

- `--iterations`: Number of iterations (for AdaBoost, Bagging, Random Forest)
- `--max_features`: For Random Forest, the number of features to consider at each split. Can be a comma-separated list (e.g., `--max_features 2,4,6`)

### Linear Regression Specific Arguments

- `--initial_learning_rate`: Initial learning rate for gradient descent methods (default: 0.01)
- `--decay_factor`: Decay factor for learning rate (default: 0.8)
- `--decay_interval`: Number of iterations between learning rate decay (default: 500)

### Bias-Variance Decomposition Specific Arguments

- `--num_runs`: Number of runs for bias-variance decomposition (default: 100)
- `--sample_size`: Sample size for each run (default: 1000)
- `--num_trees`: Number of trees in the ensemble (default: 500)
- `--bias_variance_max_features`: Number of features to consider at each split for Random Forest Bias-Variance (default: `4`)

### Examples

#### Decision Tree on Car Dataset

~~~
python main.py --algorithm decisiontree --dataset car --max_depth 6
~~~

#### Decision Tree on Bank Dataset with Unknown Values Handling

~~~
python main.py --algorithm decisiontree --dataset bank --max_depth 16 --handle_unknown y
~~~

#### AdaBoost on Bank Dataset

~~~
python main.py --algorithm adaboost --dataset bank --iterations 500
~~~

#### Bagging on Bank Dataset

~~~
python main.py --algorithm bagging --dataset bank --iterations 500
~~~

#### Random Forest on Bank Dataset

~~~
python main.py --algorithm randomforest --dataset bank --iterations 500 --max_features 4,6,8
~~~

#### Bias-Variance Decomposition for Bagging on Bank Dataset

~~~
python main.py --algorithm bagging_bias_variance --dataset bank --num_runs 100 --sample_size 1000 --num_trees 500
~~~

#### Bias-Variance Decomposition for Random Forest on Bank Dataset

~~~
python main.py --algorithm random_forest_bias_variance --dataset bank --num_runs 100 --sample_size 1000 --num_trees 500 --bias_variance_max_features 4
~~~

#### Batch Gradient Descent on Concrete Dataset

~~~
python main.py --algorithm batch_gradient_descent --dataset concrete --initial_learning_rate 0.001 --decay_factor 0.8 --decay_interval 500
~~~

#### Stochastic Gradient Descent on Concrete Dataset

~~~
python main.py --algorithm stochastic_gradient_descent --dataset concrete --initial_learning_rate 0.01 --decay_factor 0.7 --decay_interval 300
~~~

#### Analytical Solution on Concrete Dataset

~~~
python main.py --algorithm analytical_solution --dataset concrete
~~~

## Output

- **Results Directory**: Results plots such as training and test errors are saved in the `results` directory.
  
## Datasets

- **Car Evaluation Dataset** (`car`):
  - Used for classification with categorical features.
  - Contains car evaluations based on various criteria.

- **Bank Marketing Dataset** (`bank`):
  - Used for classification, includes both numerical and categorical features.
  - Contains information about bank marketing campaigns.
  - May contain 'unknown' values in some attributes.

- **Concrete Compressive Strength Dataset** (`concrete`):
  - Used for regression tasks.
  - Contains instances of concrete samples with various features influencing compressive strength.
