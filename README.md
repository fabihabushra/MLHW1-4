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
- **Perceptron Algorithms**:
  - Standard Perceptron
  - Voted Perceptron
  - Averaged Perceptron
  - Kernel Perceptron
- **Support Vector Machines (SVMs)**:
  - SVM in the Primal Domain
  - SVM in the Dual Domain
  - Nonlinear SVM using Gaussian Kernel
- **Bias-Variance Decomposition** experiments for Bagging and Random Forest

The datasets used in this project are:

- **Car Evaluation Dataset** (`car`)
- **Bank Marketing Dataset** (`bank`)
- **Concrete Compressive Strength Dataset** (`concrete`)
- **Banknote Authentication Dataset** (`bank-note`)
  
## Folder Structure

~~~
.
├── bank/               
│   ├── train.csv       
│   ├── test.csv
├── bank-note/                
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
├── Perceptron/
│   ├── standard_perceptron.py
│   ├── voted_perceptron.py
│   ├── average_perceptron.py
│   └── kernel_perceptron.py
├── SVM/
│   ├── svm_primal.py
│   ├── svm_dual.py
│   └── nonlinear_svm.py
├── results/
│   └── (output plots)
├── dataloader.py       
├── main.py

~~~

## Running the Code

There are two ways to run the code: using the `main.py` directly or executing the provided `run.sh` script.

### Option 1: Running `main.py` Directly

You can execute `main.py` to run different algorithms on various datasets by specifying command-line arguments.

**Usage:**

```bash
python main.py --algorithm <algorithm_name> --dataset <dataset_name> [additional_parameters]
```

- Replace `<algorithm_name>` with the name of the algorithm you want to run.
- Replace `<dataset_name>` with the dataset on which you want to run the algorithm.
- Optionally, add any other parameters as needed.

### Option 2: Running the Code Using `run.sh`

Alternatively, you can use the `run.sh` script located in the root directory. The `run.sh` script will automatically execute predefined commands for running various algorithms with the default settings.

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
- `standard_perceptron`
- `voted_perceptron`
- `average_perceptron`
- `kernel_perceptron`
- `svm_primal`
- `svm_dual`
- `nonlinear_svm`

### Available Datasets

- `car`
- `bank`
- `concrete`
- `bank-note`

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

### Perceptron Specific Arguments
- `--epochs`: Number of epochs for training (default: 10)

#### Support Vector Machines (SVM) Specific Arguments

- `--epochs`: Number of epochs for the SVM in the primal domain (default: 100)
- `--C`: Regularization parameter for SVM in primal and dual domain 
- `--gamma`: Gaussian kernel parameter for nonlinear SVM 

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
python main.py --algorithm randomforest --dataset bank --iterations 500 --max_features 2,4,6
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

#### Standard Perceptron on Banknote Dataset

~~~
python main.py --algorithm standard_perceptron --dataset bank-note --epochs 10
~~~

#### Voted Perceptron on Banknote Dataset

~~~
python main.py --algorithm voted_perceptron --dataset bank-note --epochs 10
~~~

#### Averaged Perceptron on Banknote Dataset

~~~
python main.py --algorithm average_perceptron --dataset bank-note --epochs 10
~~~

#### Kernel Perceptron on Banknote Dataset

```bash
python main.py --algorithm kernel_perceptron --dataset bank-note
```

#### SVM in Primal Domain on Banknote Dataset

```bash
python main.py --algorithm svm_primal --dataset bank-note --epochs 100
```

#### SVM in Dual Domain on Banknote Dataset

```bash
python main.py --algorithm svm_dual --dataset bank-note
```

#### Nonlinear SVM with Gaussian Kernel on Banknote Dataset

```bash
python main.py --algorithm nonlinear_svm --dataset bank-note
```

## Output

- **Results Directory**: Results plots such as training and test errors are saved in the `results` directory.
