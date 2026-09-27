"""
RL Mini Project 14 — Data Centre Cooling with Reinforcement Learning
=====================================================================
Agents: Q-Learning (full), Q-Learning (early stopping), Double Q-Learning, SARSA
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Hyperparameters ────────────────────────────────────────────────────────────
CFG = dict(
    episodes          = 500,
    steps_per_episode = 12,       # one step = one month
    alpha             = 0.1,      # learning rate
    gamma             = 0.9,      # discount factor
    epsilon_start     = 1.0,
    epsilon_min       = 0.01,
    epsilon_decay     = 0.995,    # multiplicative per episode
    state_bins        = 10,
    actions           = 5,
    early_stop_patience = 30,
    eval_runs         = 100,      # episodes averaged for evaluation
)
print("Config loaded:", CFG)


# ══════════════════════════════════════════════════════════════════════════════
# 1 · Environment
# ══════════════════════════════════════════════════════════════════════════════

class DataCenterEnvironment:
    """
    Simulated data-centre with realistic thermal physics.

    State  : [temperature, users, data_rate]
    Actions: 5 discrete cooling adjustments [-3, -1.5, 0, +1.5, +3] °C
    Reward : energy saved vs. rule-based baseline, minus out-of-band penalty
    """

    OPTIMAL_LOW  = 18.0
    OPTIMAL_HIGH = 24.0
    MIN_TEMP     = -20.0
    MAX_TEMP     =  80.0
    ACTION_DELTA = [-3, -1.5, 0, 1.5, 3]

    MONTHLY_ATMO = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0,
                    30.0, 28.0, 22.0, 15.0, 10.0, 5.0]

    def __init__(self, seed=None):
        self._rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.atmospheric_temperature = self.MONTHLY_ATMO[0]
        self.temperature   = float(self.atmospheric_temperature)
        self.users         = self._rng.randint(10, 101)
        self.data_rate     = self._rng.randint(20, 301)
        self.total_energy_ai   = 0.0
        self.total_energy_noai = 0.0
        return self._get_state()

    def _get_state(self):
        return np.array([self.temperature, self.users, self.data_rate], dtype=np.float32)

    def update_env(self, action: int, month: int):
        """Step the environment one month forward."""
        delta_temp = self.ACTION_DELTA[action]

        # Physics: load heats the server, atmosphere drifts it
        load_heat  = (self.users / 100) * 2.0 + (self.data_rate / 300) * 1.5
        atmo_drift = 0.1 * (self.atmospheric_temperature - self.temperature)

        self.temperature += delta_temp + load_heat + atmo_drift

        # Update stochastic env variables
        self.atmospheric_temperature = self.MONTHLY_ATMO[month % 12]
        self.users     = self._rng.randint(10, 101)
        self.data_rate = self._rng.randint(20, 301)

        # Energy accounting
        energy_ai   = abs(delta_temp)
        energy_noai = abs(self.temperature - 21.0)

        self.total_energy_ai   += energy_ai
        self.total_energy_noai += energy_noai

        # Reward: savings − out-of-band penalty
        reward = energy_noai - energy_ai
        if not (self.OPTIMAL_LOW <= self.temperature <= self.OPTIMAL_HIGH):
            penalty = abs(self.temperature - (self.OPTIMAL_LOW + self.OPTIMAL_HIGH) / 2) * 0.5
            reward -= penalty

        self.temperature = np.clip(self.temperature, self.MIN_TEMP, self.MAX_TEMP)
        return self._get_state(), reward


# Sanity check
_e = DataCenterEnvironment(seed=0)
_s = _e.reset()
_ns, _r = _e.update_env(2, 1)
print(f"State shape: {_s.shape}, sample reward: {_r:.3f}  ✓")


# ══════════════════════════════════════════════════════════════════════════════
# 2 · Agents
# ══════════════════════════════════════════════════════════════════════════════

class QLearningAgent:
    """
    Tabular Q-Learning with epsilon-greedy exploration.

    Parameters
    ----------
    double : bool
        If True, use Double Q-Learning (two Q-tables, alternating updates).
    """

    def __init__(self, state_bins=10, actions=5,
                 alpha=0.1, gamma=0.9, epsilon=1.0,
                 double=False, seed=None):
        self.state_bins = state_bins
        self.actions    = actions
        self.alpha      = alpha
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.double     = double
        self._rng       = np.random.RandomState(seed)

        self.q_table  = np.zeros((state_bins, state_bins, state_bins, actions))
        self.q_table2 = np.zeros_like(self.q_table) if double else None

    def discretize(self, state):
        temp, users, data = state
        t = int((temp + 20) / 100 * self.state_bins)
        u = int(users        / 100 * self.state_bins)
        d = int(data         / 300 * self.state_bins)
        clip = lambda x: np.clip(x, 0, self.state_bins - 1)
        return clip(t), clip(u), clip(d)

    def select_action(self, state):
        if self._rng.rand() < self.epsilon:
            return self._rng.randint(self.actions)
        s = self.discretize(state)
        q = self.q_table[s] + (self.q_table2[s] if self.double else 0)
        return int(np.argmax(q))

    def update(self, state, action, reward, next_state):
        s  = self.discretize(state)
        ns = self.discretize(next_state)

        if self.double and self._rng.rand() < 0.5:
            best_a = int(np.argmax(self.q_table[ns]))
            target = reward + self.gamma * self.q_table2[ns][best_a]
            self.q_table2[s][action] += self.alpha * (target - self.q_table2[s][action])
        else:
            best_next = np.max(self.q_table[ns])
            target    = reward + self.gamma * best_next
            self.q_table[s][action] += self.alpha * (target - self.q_table[s][action])

print("QLearningAgent defined  ✓")


class SARSAAgent(QLearningAgent):
    """SARSA: on-policy TD control."""

    def update(self, state, action, reward, next_state, next_action=None):
        s  = self.discretize(state)
        ns = self.discretize(next_state)

        if next_action is None:
            next_action = self.select_action(next_state)

        next_q = self.q_table[ns][next_action]
        target = reward + self.gamma * next_q
        self.q_table[s][action] += self.alpha * (target - self.q_table[s][action])
        return next_action

print("SARSAAgent defined  ✓")


# ══════════════════════════════════════════════════════════════════════════════
# 3 · Training
# ══════════════════════════════════════════════════════════════════════════════

def train(agent, env_fn, cfg, early_stopping=False, desc=""):
    """
    Train *agent* for cfg['episodes'] episodes.

    Returns
    -------
    rewards, energy_ai, energy_noai, epsilons, stopped_at
    """
    rewards, energy_ai, energy_noai, epsilons = [], [], [], []
    best_reward, patience_ctr, best_q = -np.inf, 0, None
    stopped_at = None
    is_sarsa   = isinstance(agent, SARSAAgent)

    for ep in tqdm(range(cfg['episodes']), desc=desc or type(agent).__name__, leave=True):
        env   = env_fn()
        state = env.reset()
        total_reward = 0

        if is_sarsa:
            action = agent.select_action(state)

        for step in range(cfg['steps_per_episode']):
            if is_sarsa:
                next_state, reward = env.update_env(action, step)
                next_action        = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
            else:
                action             = agent.select_action(state)
                next_state, reward = env.update_env(action, step)
                agent.update(state, action, reward, next_state)
                state = next_state

            total_reward += reward

        rewards.append(total_reward)
        energy_ai.append(env.total_energy_ai)
        energy_noai.append(env.total_energy_noai)
        epsilons.append(agent.epsilon)

        if early_stopping:
            if total_reward > best_reward:
                best_reward, patience_ctr = total_reward, 0
                best_q = agent.q_table.copy()
                with open("best_q_table.pkl", "wb") as f:
                    pickle.dump(best_q, f)
            else:
                patience_ctr += 1
            if patience_ctr >= cfg['early_stop_patience']:
                stopped_at = ep
                agent.q_table = best_q
                print(f"\n  ⏹  Early stopping at episode {ep}")
                break

        agent.epsilon = max(cfg['epsilon_min'], agent.epsilon * cfg['epsilon_decay'])

    return rewards, energy_ai, energy_noai, epsilons, stopped_at


env_fn = lambda: DataCenterEnvironment(seed=None)
print("train() ready  ✓")

# ── Run all four agents ────────────────────────────────────────────────────────
agent_full = QLearningAgent(**{k: CFG[k] for k in
    ['state_bins','actions','alpha','gamma']}, epsilon=CFG['epsilon_start'], seed=SEED)
r_full, ea_full, en_full, eps_full, _ = train(
    agent_full, env_fn, CFG, early_stopping=False, desc="Q-Learning (full)")

agent_es = QLearningAgent(**{k: CFG[k] for k in
    ['state_bins','actions','alpha','gamma']}, epsilon=CFG['epsilon_start'], seed=SEED)
r_es, ea_es, en_es, eps_es, stopped_ep = train(
    agent_es, env_fn, CFG, early_stopping=True, desc="Q-Learning (early stop)")

agent_dq = QLearningAgent(**{k: CFG[k] for k in
    ['state_bins','actions','alpha','gamma']}, epsilon=CFG['epsilon_start'],
    double=True, seed=SEED)
r_dq, ea_dq, en_dq, eps_dq, _ = train(
    agent_dq, env_fn, CFG, early_stopping=False, desc="Double Q-Learning")

agent_sarsa = SARSAAgent(**{k: CFG[k] for k in
    ['state_bins','actions','alpha','gamma']}, epsilon=CFG['epsilon_start'], seed=SEED)
r_sarsa, ea_sarsa, en_sarsa, eps_sarsa, _ = train(
    agent_sarsa, env_fn, CFG, early_stopping=False, desc="SARSA")

print("\nAll agents trained  ✓")


# ══════════════════════════════════════════════════════════════════════════════
# 4 · Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_agent(agent, env_fn, runs=100):
    """Run *runs* deterministic (greedy) evaluation episodes."""
    ai_list, noai_list, temp_trajs = [], [], []

    for _ in range(runs):
        env   = env_fn()
        state = env.reset()
        traj  = [env.temperature]

        for step in range(12):
            s      = agent.discretize(state)
            action = int(np.argmax(agent.q_table[s]))
            state, _ = env.update_env(action, step)
            traj.append(env.temperature)

        ai_list.append(env.total_energy_ai)
        noai_list.append(env.total_energy_noai)
        temp_trajs.append(traj)

    return np.array(ai_list), np.array(noai_list), np.array(temp_trajs)


results = {}
for name, ag in [("Q-Learning Full",  agent_full),
                 ("Q-Learning ES",    agent_es),
                 ("Double Q",         agent_dq),
                 ("SARSA",            agent_sarsa)]:
    ai, noai, trajs = evaluate_agent(ag, env_fn, runs=CFG['eval_runs'])
    saving = 100 * (noai - ai) / noai
    results[name] = dict(ai=ai, noai=noai, trajs=trajs, saving=saving)

print(f"\n{'Method':<22} {'AI Energy':>12} {'No-AI Energy':>14} {'Saved %':>10}")
print("─" * 62)
for name, r in results.items():
    print(f"{name:<22} {r['ai'].mean():>9.2f}±{r['ai'].std():>4.1f}"
          f"  {r['noai'].mean():>10.2f}±{r['noai'].std():>4.1f}"
          f"  {r['saving'].mean():>8.2f}±{r['saving'].std():>4.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 5 · Visualisations
# ══════════════════════════════════════════════════════════════════════════════

window = 20

# 5a · Reward curves
fig, ax = plt.subplots(figsize=(14, 4))
for label, r, style in [
    ("Q-Learning Full",  r_full,  '-'),
    ("Q-Learning ES",    r_es,    '--'),
    ("Double Q",         r_dq,    '-.'),
    ("SARSA",            r_sarsa, ':'),
]:
    ax.plot(r, alpha=0.3, linestyle=style)
    ma = np.convolve(r, np.ones(window)/window, mode='valid')
    ax.plot(ma, label=label, linestyle=style, linewidth=2)

ax.set_title("Reward per Episode (moving avg)")
ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# 5b · Energy savings bar chart
fig, ax = plt.subplots(figsize=(9, 5))
names  = list(results.keys())
means  = [results[n]['saving'].mean() for n in names]
stds   = [results[n]['saving'].std()  for n in names]
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor='k')
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{m:.1f}%', ha='center', va='bottom', fontweight='bold')

ax.set_title(f"Mean Energy Saved vs. Rule-Based Controller ({CFG['eval_runs']} eval runs)")
ax.set_ylabel("Energy Saved (%)")
ax.set_ylim(0, max(means) * 1.3)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.show()

# 5c · Temperature trajectory
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan']
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)

for ax, (name, r) in zip(axes.flat, results.items()):
    trajs  = r['trajs']
    mean_t = trajs.mean(axis=0)
    std_t  = trajs.std(axis=0)

    ax.fill_between(range(13), mean_t - std_t, mean_t + std_t, alpha=0.25, label='±1 std')
    ax.plot(mean_t, linewidth=2, label='Mean temp')
    ax.axhspan(18, 24, alpha=0.12, color='green', label='Optimal band')
    ax.axhline(18, color='green', linestyle='--', linewidth=0.8)
    ax.axhline(24, color='green', linestyle='--', linewidth=0.8)
    ax.set_title(name); ax.set_xticks(range(13)); ax.set_xticklabels(months, fontsize=8)
    ax.set_ylabel("Temperature (°C)"); ax.set_xlabel("Month")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle("Server Temperature Trajectory during Evaluation", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

# 5d · Q-table heatmap
q  = agent_dq.q_table
SB = CFG['state_bins']
q_temp = q.mean(axis=(1, 2))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

im = axes[0].imshow(q_temp, aspect='auto', cmap='RdYlGn', origin='lower')
axes[0].set_title("Q-table Heatmap (avg over users/data bins)\nDouble Q-Learning")
axes[0].set_xlabel("Action  [−3  −1.5  0  +1.5  +3 °C]")
axes[0].set_ylabel("Temperature bin  (0=cold  9=hot)")
axes[0].set_xticks(range(5)); axes[0].set_xticklabels(['-3','-1.5','0','+1.5','+3'])
plt.colorbar(im, ax=axes[0], label='Q-value')

greedy       = np.argmax(q_temp, axis=1)
action_labels = ['-3','-1.5','0','+1.5','+3']
colors_act    = ['#1f77b4','#aec7e8','#98df8a','#ffbb78','#d62728']
axes[1].barh(range(SB), [1]*SB, color=[colors_act[a] for a in greedy], edgecolor='k')
axes[1].set_title("Greedy Action per Temperature Bin\nDouble Q-Learning")
axes[1].set_xlabel("Preferred action"); axes[1].set_ylabel("Temperature bin")
axes[1].set_yticks(range(SB)); axes[1].set_yticklabels([f"bin {i}" for i in range(SB)])

legend_handles = [Patch(color=colors_act[i], label=action_labels[i]) for i in range(5)]
axes[1].legend(handles=legend_handles, title="Action (°C)", loc='lower right', fontsize=8)
plt.tight_layout(); plt.show()

# 5e · Cumulative energy savings over training
fig, ax = plt.subplots(figsize=(11, 4))
for label, ea, en, style in [
    ("Q-Learning Full",  ea_full,  en_full,  '-'),
    ("Q-Learning ES",    ea_es,    en_es,    '--'),
    ("Double Q",         ea_dq,    en_dq,    '-.'),
    ("SARSA",            ea_sarsa, en_sarsa, ':'),
]:
    saved_pct = [(n - a) / n * 100 if n > 0 else 0 for a, n in zip(ea, en)]
    ax.plot(saved_pct, label=label, linestyle=style, linewidth=2)

ax.set_title("Cumulative Energy Savings (%) over Training Episodes")
ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Savings (%)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
print("\nAll visualisations complete  ✓")


# ══════════════════════════════════════════════════════════════════════════════
# 6 · Save / Load Best Model
# ══════════════════════════════════════════════════════════════════════════════

best_name = max(results, key=lambda n: results[n]['saving'].mean())
best_agent_map = {
    "Q-Learning Full": agent_full,
    "Q-Learning ES":   agent_es,
    "Double Q":        agent_dq,
    "SARSA":           agent_sarsa,
}
best_agent = best_agent_map[best_name]

with open("best_q_table.pkl", "wb") as f:
    pickle.dump(best_agent.q_table, f)

print(f"Best agent: {best_name}  "
      f"(mean saving {results[best_name]['saving'].mean():.2f}%)")
print("Model saved to best_q_table.pkl  ✓")

# Verify
with open("best_q_table.pkl", "rb") as f:
    loaded_q = pickle.load(f)

assert loaded_q.shape == best_agent.q_table.shape
print("Model loaded and shape verified:", loaded_q.shape, " ✓")
