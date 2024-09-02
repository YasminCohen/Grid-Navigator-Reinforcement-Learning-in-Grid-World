# Grid World - Solving with Reinforcement Learning Techniques

## Introduction

The Grid World problem is a classic reinforcement learning (RL) task where an agent must navigate through a grid to maximize cumulative rewards. The grid contains various cells with rewards and penalties, and the agent's movement is probabilistic, meaning that it may not always move in the direction intended. This project implements and compares three different approaches to solve the Grid World problem, aiming to find the optimal policy for the agent.

## Problem Description

### Grid World Setup

- **Grid Dimensions (W x H):** The grid is defined by its width (`W`) and height (`H`).
- **Rewards and Penalties (L):** Specific cells in the grid contain rewards (positive values) or penalties (negative values). These are provided as a list of tuples, with each tuple containing the coordinates of the cell and the associated value.
  - Example: `L = [(x1, y1, r1), (x2, y2, r2), ...]` where `(xi, yi)` is the position and `ri` is the reward or penalty at that position.
- **Success Probability (p):** The probability that the agent successfully moves in the intended direction. For example, if `p = 0.8`, there is an 80% chance that the agent will move as intended, and a 20% chance that it will move in a perpendicular direction.
- **Step Cost (r):** Each move incurs a cost, which can be positive or negative. This reflects the penalty or reward for taking a step.
- **Discount Factor (γ):** The discount factor determines the importance of future rewards. A γ close to 1 makes future rewards more important, while a γ close to 0 makes immediate rewards more significant. In this project, γ is set to 0.5.

### Goal

The primary goal is to implement and evaluate three different reinforcement learning approaches to derive an optimal policy for the Grid World problem:

1. **Value Iteration (MDP-based):** Using dynamic programming techniques, specifically Bellman equations, to compute the optimal value function and policy.
2. **Model-Based Reinforcement Learning:** Learning a model of the environment (transition probabilities and rewards) and using this model to compute the optimal policy.
3. **Model-Free Reinforcement Learning (SARSA):** Learning the optimal policy directly from interactions with the environment, without explicitly modeling the environment’s dynamics.

## Project Structure

The project is organized into the following key files:

- **`main.py`:** This is the main script that orchestrates the execution of experiments on various grid configurations. It calls the appropriate functions from the other scripts to calculate policies and compare results.
- **`MDP.py`:** Contains the implementation of the value iteration algorithm. This script is responsible for computing the optimal policy using the Markov Decision Process (MDP) framework.
- **`modelFree.py`:** Implements the SARSA algorithm, which is a model-free reinforcement learning method. This script learns the policy directly from the agent's experiences.
- **`baseModelRL.py`:** Implements a model-based reinforcement learning approach, where the agent first learns the environment's transition model and then derives the optimal policy.
- **`gridResults.xlsx`:** An Excel file where the results from various grid configurations are stored, including value function differences and policy comparisons.

## Implementation Details

### 1. Value Iteration (`MDP.py`)

- **Value Function Initialization:** The value function is initialized based on the rewards and penalties specified in the grid.
- **Bellman Update:** The value function is iteratively updated using the Bellman optimality equation until convergence, which is determined by a threshold (`epsilon`).
- **Policy Extraction:** Once the value function converges, the optimal policy is derived by choosing the action that maximizes the expected utility for each state.

### 2. Model-Based Reinforcement Learning (`baseModelRL.py`)

- **Environment Modeling:** The agent interacts with the grid to learn the transition probabilities and expected rewards for each state-action pair.
- **Policy Improvement:** Using the learned model, the agent performs policy iteration, where it alternates between policy evaluation (computing the value function for the current policy) and policy improvement (updating the policy to be greedy with respect to the current value function).

### 3. Model-Free Reinforcement Learning (`modelFree.py`)

- **SARSA Algorithm:** The agent follows an epsilon-greedy policy, which balances exploration and exploitation. It updates the Q-values based on the observed rewards and the value of subsequent states.
- **Epsilon-Greedy Exploration:** The agent occasionally explores random actions to discover potentially better policies.
- **Policy Extraction:** Once the Q-values have been learned, the optimal policy is derived by selecting the action with the highest Q-value for each state.

## Running the Experiments

To run the experiments:

1. Ensure all the required Python packages are installed: NumPy and Pandas.
2. Execute `main.py` in your Python environment. This will run the grid experiments using predefined configurations and save the results in `gridResults.xlsx`.

### Example Command:

```bash
python main.py
