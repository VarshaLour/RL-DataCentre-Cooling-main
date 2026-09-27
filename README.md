# 🧊 RL Data Centre Cooling

A Streamlit-based interactive application that trains and compares **Reinforcement Learning** agents for optimizing data centre cooling energy consumption.

## Agents

| Agent | Type | Key Feature |
|---|---|---|
| **Q-Learning (full)** | Off-policy TD | Trains for all episodes |
| **Q-Learning (early stop)** | Off-policy TD | Stops when no improvement |
| **Double Q-Learning** | Off-policy TD | Two Q-tables to reduce overestimation |
| **SARSA** | On-policy TD | Updates use the actual next action taken |

## How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Features

- Configurable hyperparameters via sidebar
- Real-time training progress bars
- Reward curves, energy savings bar charts
- Temperature trajectory visualizations
- Q-table heatmaps
- Download trained Q-table as `.pkl`
