# First-Order vs Second-Order Optimization Methods

This repository contains a comparison of **first-order**, **accelerated first-order**, **second-order**, and **quasi-Newton** optimization methods for the same convex machine learning problem: **L2-regularized logistic regression**.

The project compares:

* **Gradient Descent (GD)**
* **Nesterov Accelerated Gradient (NAG)**
* **Newton's Method**
* **BFGS**

The experiments are performed on the **LIBSVM Adult a9a** binary classification dataset. The goal is to study how different optimization methods behave in terms of convergence speed, runtime per iteration, total runtime, numerical stability, scalability with problem size, and final classification performance.

---

## Project Overview

Optimization problems in machine learning can often be solved using either first-order or second-order methods. First-order methods use gradient information only, while second-order methods use curvature information through the Hessian matrix.

This project studies the main trade-off between these methods:

* First-order methods usually have cheaper iterations but may require many iterations.
* Second-order methods usually converge in fewer iterations but have a higher cost per iteration.
* Quasi-Newton methods approximate curvature information without explicitly computing the exact Hessian.

The implementation focuses on a fair comparison by applying all methods to the same objective function, dataset, and evaluation pipeline.

---

## Problem Formulation

The selected optimization problem is **binary logistic regression with L2 regularization**. For training samples `(x_i, y_i)`, where `y_i ∈ {0, 1}`, the model predicts:

```text
p_i = sigmoid(w^T x_i)
```

The optimized objective is the L2-regularized logistic loss:

```text
J(w) = (1/N) Σ [ log(1 + exp(w^T x_i)) - y_i w^T x_i ]
       + (λ / 2) ||w_reg||²
```

A bias term is added to the feature matrix, but the bias term is **not regularized**.

---

## Dataset

The project uses the **LIBSVM Adult a9a** binary classification dataset.

| Split                      | Samples |
| -------------------------- | ------: |
| Training set               |  32,561 |
| Test set                   |  16,281 |
| Total                      |  48,842 |
| Features after adding bias |     124 |

The dataset is suitable for this project because it is larger than small toy datasets and allows clearer comparison of runtime, convergence behavior, and scalability.

---

## Methods Compared

| Method                        | Type                    | Information Used                       | Purpose                                   |
| ----------------------------- | ----------------------- | -------------------------------------- | ----------------------------------------- |
| Gradient Descent              | First-order             | Gradient only                          | Baseline first-order method               |
| Nesterov Accelerated Gradient | Accelerated first-order | Gradient and momentum                  | Compare acceleration with standard GD     |
| Newton's Method               | Second-order            | Gradient and exact Hessian             | Exact curvature-based method              |
| BFGS                          | Quasi-Newton            | Gradient-based curvature approximation | Practical approximate second-order method |

---

## Main Optimization Results

| Method                        | Iterations | Final Loss | Final Gradient Norm | Total Time (ms) | Mean/Iteration (ms) |
| ----------------------------- | ---------: | ---------: | ------------------: | --------------: | ------------------: |
| Gradient Descent              |       5000 |   0.332838 |          2.4270e-04 |        71298.35 |               9.396 |
| Nesterov Accelerated Gradient |       5000 |   0.332713 |          1.2022e-06 |       120601.18 |               9.598 |
| Newton's Method               |          7 |   0.332713 |          2.0956e-07 |          643.29 |              87.240 |
| BFGS                          |        200 |   0.332713 |          8.1398e-06 |         5332.09 |              26.594 |

Newton's Method required the fewest iterations because it uses exact curvature information. Gradient Descent and Nesterov Accelerated Gradient had cheaper iterations but required many more updates. BFGS provided a practical middle ground by approximating curvature information.

---

## Classification Results

| Method                        | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ----------------------------- | -------: | --------: | -----: | -------: | ------: |
| Gradient Descent              |   0.8510 |    0.7311 | 0.5840 |   0.6493 |  0.9025 |
| Nesterov Accelerated Gradient |   0.8513 |    0.7313 | 0.5858 |   0.6505 |  0.9026 |
| Newton's Method               |   0.8513 |    0.7313 | 0.5858 |   0.6505 |  0.9026 |
| BFGS                          |   0.8513 |    0.7313 | 0.5858 |   0.6505 |  0.9026 |

The classification metrics are almost identical because all methods optimize the same logistic regression objective. The main difference is not the final predictive performance, but the optimization path and computational cost needed to reach the solution.

---

## Plots

> The images are expected to be inside a folder named `plot/` in this repository.

### Iterations Used

![Iterations Used](plot/cmp_01_iterations.png)

### Total Training Time

![Total Training Time](plot/cmp_02_total_time.png)

### Runtime per Iteration

![Runtime per Iteration](plot/cmp_03_iter_time.png)

### Final Objective Value

![Final Objective Value](plot/cmp_04_final_loss.png)

### Objective Value vs Iteration

![Objective Value vs Iteration](plot/cmp_05_loss_vs_iteration.png)

### Gradient Norm vs Iteration

![Gradient Norm vs Iteration](plot/cmp_06_grad_norm_vs_iteration.png)

### Objective Value vs Time

![Objective Value vs Time](plot/cmp_07_loss_vs_time.png)

### Scalability: Total Runtime vs Problem Size

![Scalability Total Runtime](plot/cmp_08_scalability_total_runtime.png)

### Scalability: Cost per Iteration vs Problem Size

![Scalability Iteration Time](plot/cmp_09_scalability_iter_time.png)

---

## Scalability Results

The scalability experiment uses training subsets of size 1000, 5000, 10000, 20000, and 32561 samples.

| Samples | Method | Iterations | Total Time (ms) | Mean/Iter (ms) | Final Loss |
| ------: | ------ | ---------: | --------------: | -------------: | ---------: |
|    1000 | GD     |       1500 |          365.28 |          0.137 |   0.310596 |
|    1000 | NAG    |       1500 |          568.41 |          0.136 |   0.309624 |
|    1000 | Newton |          8 |           21.86 |          2.559 |   0.309609 |
|    1000 | BFGS   |         80 |          150.40 |          1.868 |   0.309879 |
|    5000 | GD     |       1500 |         2171.29 |          0.888 |   0.330993 |
|    5000 | NAG    |       1500 |         6185.14 |          1.597 |   0.330622 |
|    5000 | Newton |          7 |          129.87 |         17.763 |   0.330613 |
|    5000 | BFGS   |         80 |          291.91 |          3.624 |   0.330782 |
|   10000 | GD     |       1500 |         4325.41 |          1.801 |   0.336306 |
|   10000 | NAG    |       1500 |         9496.64 |          2.434 |   0.335925 |
|   10000 | Newton |          7 |          167.01 |         22.424 |   0.335914 |
|   10000 | BFGS   |         80 |          568.75 |          7.065 |   0.336110 |
|   20000 | GD     |       1500 |        12810.30 |          5.548 |   0.334998 |
|   20000 | NAG    |       1500 |        19296.78 |          5.038 |   0.334678 |
|   20000 | Newton |          7 |          345.76 |         46.575 |   0.334667 |
|   20000 | BFGS   |         80 |         1672.21 |         20.619 |   0.334848 |
|   32561 | GD     |       1500 |        22121.36 |          9.741 |   0.333060 |
|   32561 | NAG    |       1500 |        36502.82 |          9.675 |   0.332725 |
|   32561 | Newton |          7 |          575.30 |         77.573 |   0.332713 |
|   32561 | BFGS   |         80 |         2194.59 |         27.266 |   0.332906 |

---

## Key Findings

* **Gradient Descent** is simple and cheap per iteration, but it requires many iterations.
* **Nesterov Accelerated Gradient** improves over standard Gradient Descent by reaching a lower loss and smaller gradient norm within the same iteration budget.
* **Newton's Method** converges in very few iterations because it uses exact Hessian information, but each iteration is more expensive.
* **BFGS** provides a strong practical compromise by approximating curvature information.
* On this dataset, Newton's Method is very efficient because the number of features is moderate.
* The scalability experiment shows why exact second-order methods may become expensive for larger or higher-dimensional problems.

---

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

---

## Running the Notebook

Open the notebook:

```bash
jupyter notebook First-Order vs Second-Order Optimization.ipynb
```

or run it in Google Colab.

The notebook downloads the LIBSVM Adult a9a dataset, trains the optimization methods, evaluates the results, and saves the plots in the `plot/` folder.

---

## Dependencies

The main Python packages used are:

* NumPy
* Pandas
* Matplotlib
* scikit-learn
* SciPy
* Jupyter

---

## Authors

* Nemer Saab — `nas83@mail.aub.edu`
* Mohammed Haydar — `mhh87@mail.aub.edu`
* Issa Al Sabeh — `iha20@mail.aub.edu`

American University of Beirut

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
