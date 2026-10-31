# Intro to Machine Learning — Assignments

Coursework from **CSE 574 Introduction to Machine Learning** (University at Buffalo, Fall 2025),
built from scratch in NumPy and PyTorch: classical models, neural networks, CNNs and reinforcement learning.
Each assignment was done in a team of two; teammates are credited below and inside every notebook.

## Highlights

| Result | Where |
|--------|-------|
| **91.9% test accuracy** (F1 0.918) with a VGG-13 built from scratch on a 36-class image dataset (10,080 held-out test images) | [A2 · part 4](a2-neural-networks-and-cnns/part4_vgg13.ipynb) |
| **91.1% test accuracy** (F1 0.909) with a custom CNN (BatchNorm + dropout) on the same data | [A2 · part 3](a2-neural-networks-and-cnns/part3_custom_cnn.ipynb) |
| ResNet-34 implemented from scratch (21.3 M params) and trained on Flowers-102 with only 1,020 training images | [A2 · bonus](a2-neural-networks-and-cnns/bonus_resnet34_from_scratch.ipynb) |
| CNN interpretability: feature maps, activations and learned filters | [A2 · bonus](a2-neural-networks-and-cnns/bonus_cnn_interpretability.ipynb) |
| Logistic regression from scratch: **92.8% accuracy** on Palmer Penguins | [A1 · part 2](a1-regression-and-classification/part2_logistic_regression.ipynb) |
| Custom Gymnasium environment + n-step Double Q-learning vs SARSA | [A3](a3-reinforcement-learning) |

## A1 — Regression and classification from scratch (NumPy)
*Team: Pranay Lachuluri, Odra Bira*

- **Part 1:** cleaning, encoding, outlier handling and EDA on Penguins, Diamonds and a NY breeding-bird atlas dataset.
- **Part 2:** binary logistic regression with gradient descent and hyperparameter search (accuracy 0.928, precision 0.969, F1 0.925).
- **Part 3:** linear regression via OLS and ridge regression with a λ sweep on Diamonds (test R² 0.718); discussion of the bias–variance trade-off.
- **Part 4:** elastic-net regression with Xavier initialization, gradient descent and two early-stopping criteria, compared against ridge and lasso.
- **Bonus:** multinomial softmax regression with L2 regularization (91.8% on Penguins) and logistic regression on Buffalo 311 service-request open data.

## A2 — Neural networks and CNNs (PyTorch)
*Team: Pranay Lachuluri, teammate vpamulap (UB)*

- **Part 1:** fully connected network on a 766-row, 7-feature tabular dataset, with zero-value imputation, scaling and ROC analysis (75.7% test accuracy).
- **Part 2:** tuning dropout, batch size, learning rate, optimizers and initializations, with accuracy and training-time comparisons.
- **Part 3:** custom CNN on 64×64 grayscale images across 36 classes — 91.1% test accuracy, confusion matrix, per-class ROC.
- **Part 4:** VGG-13 implemented layer by layer — 91.9% test accuracy (precision 0.920, recall 0.919).
- **Bonus:** ResNet-34 with residual BasicBlocks and data augmentation, trained from random initialization on Flowers-102 (40.6% top-1 across 102 classes with 10 images per class), and CNN interpretability visualizations.

## A3 — Reinforcement learning (Gymnasium)
*Team: Pranay Lachuluri, Sai Rohith Bharadwaj Ramavarapu (sramavar)*

- **Part 1:** designed `TreasureGridEnv`, a 5×5 Gymnasium environment with traps, mud tiles, a bonus tile and a goal, with a custom reward structure and renderer.
- **Part 3:** implemented n-step Double Q-learning (two Q-tables to reduce overestimation) for n = 1…5 and a SARSA baseline.
  The greedy policies for n = 1 and n = 4 reach the goal via the bonus tile (reward 88 every episode); n = 1 had the best
  average reward over the last 50 training episodes (52.5).

## Running the notebooks
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision gymnasium
jupyter notebook
```
Course datasets are not redistributed. Notebooks expect them next to the notebook
(e.g. `penguins.csv`, `diamond.csv`, `cnn_dataset/`); Flowers-102 is available via `torchvision.datasets.Flowers102`.
