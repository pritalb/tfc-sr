# Task-Focused Consolidation with Spaced Recall: Making Neural Networks learn like college students

**Prital Bamnodkar**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Deep Neural Networks often suffer from a critical limitation known as Catastrophic Forgetting, where performance on past tasks degrades after learning new ones. This paper introduces a novel continual learning approach inspired by human learning strategies like Active Recall, Deliberate Practice and Spaced Repetition, named **Task-Focused Consolidation with Spaced Replay (TFC-SR)**. TFC-SR enhances the standard experience replay with a mechanism we termed the **Active Recall Probe**. It is a periodic, task-aware evaluation of the model’s memory that stabilizes the representations of past knowledge. We test TFC-SR on the Split MNIST and the Split CIFAR-100 benchmarks against leading regularization-based and replay-based baselines. Our results show that TFC-SR performs significantly better than these methods. For instance, on the Split CIFAR-100, it achieves a final accuracy of **13.17%** compared to standard replay’s 7.40%. We demonstrate that this advantage comes from the stabilizing effect of the probe itself, and not from the difference in replay volume. Additionally, we analyze the trade-off between memory size and performance and show that while TFC-SR performs better in memory-constrained environments, higher replay volume is still more effective when available memory is abundant. We conclude that TFC-SR is a robust and efficient approach, highlighting the importance of integrating active memory retrieval mechanisms into continual learning systems.

## Key Results

The primary finding of this work is that TFC-SR, our proposed method, consistently outperforms strong continual learning baselines on the challenging Split CIFAR-100 benchmark. The "Active Recall Probe" mechanism provides a significant performance boost over standard replay methods.

![Main Comparison Plot](figures/cifar_tfc_vs_all.png)
*Figure: Final performance comparison on Split CIFAR-100.*

## Setup and Installation

This project was developed using Python 3.10 and PyTorch. All dependencies are listed in the `requirements.txt` file.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pritalb/tfc-sr.git
    cd tfc-sr
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (e.g., venv or conda).
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

The repository is structured to easily reproduce all experiments and figures from the paper.

*   **To run all experiments from scratch:** Open and run the cells in `experiments_mnist.ipynb` and `experiments_cifar100.ipynb`. Please note that training on the CIFAR-100 benchmark is computationally intensive and requires a GPU.

*   **To visualize the final results:** Open and run the `plots_and_results.ipynb` notebook. This notebook loads the pre-computed results data from the `results/` directory and generates all the figures presented in the paper.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.