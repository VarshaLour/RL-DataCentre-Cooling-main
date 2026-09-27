"""
Streamlit Interface — RL Data Centre Cooling
============================================
Run with:  streamlit run streamlit_app.py
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import io
import streamlit as st
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RL Data Centre Cooling",
    page_icon="🧊",
    layout="wide",
)

st.title("🧊 RL Data Centre Cooling")
st.markdown(
    "Train **Q-Learning**, **Double Q-Learning**, and **SARSA** agents "
    "to minimise cooling energy in a simulated data centre."
)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Hyperparameters")

episodes           = st.sidebar.slider("Episodes",            100, 2000, 500, 50)
steps_per_episode  = st.sidebar.slider("Steps per episode",   6,   24,   12,  1)
alpha              = st.sidebar.slider("Learning rate α",     0.01, 0.5, 0.1, 0.01)
gamma              = st.sidebar.slider("Discount factor γ",   0.5,  1.0, 0.9, 0.01)
epsilon_start      = st.sidebar.slider("ε start",             0.5,  1.0, 1.0, 0.05)
epsilon_min        = st.sidebar.slider("ε min",               0.001,0.1, 0.01,0.001)
epsilon_decay      = st.sidebar.slider("ε decay",             0.990,0.999,0.995,0.001)
state_bins         = st.sidebar.slider("State bins",          5,   20,   10,  1)
early_stop_patience= st.sidebar.slider("Early-stop patience", 10,  100,  30,  5)
eval_runs          = st.sidebar.slider("Eval runs",           10,  200,  100, 10)
seed               = st.sidebar.number_input("Random seed",   value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Agents to train")
run_qfull  = st.sidebar.checkbox("Q-Learning (full)",         value=True)
run_qes    = st.sidebar.checkbox("Q-Learning (early stop)",   value=True)
run_dq     = st.sidebar.checkbox("Double Q-Learning",         value=True)
run_sarsa  = st.sidebar.checkbox("SARSA",                     value=True)

CFG = dict(
    episodes=episodes, steps_per_episode=steps_per_episode,
    alpha=alpha, gamma=gamma, epsilon_start=epsilon_start,
    epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
    state_bins=state_bins, actions=5,
    early_stop_patience=early_stop_patience, eval_runs=eval_runs,
)

# ══════════════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════════════

class DataCenterEnvironment:
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
        delta_temp = self.ACTION_DELTA[action]
        load_heat  = (self.users / 100) * 2.0 + (self.data_rate / 300) * 1.5
        atmo_drift = 0.1 * (self.atmospheric_temperature - self.temperature)
        self.temperature += delta_temp + load_heat + atmo_drift
        self.atmospheric_temperature = self.MONTHLY_ATMO[month % 12]
        self.users     = self._rng.randint(10, 101)
        self.data_rate = self._rng.randint(20, 301)
        energy_ai   = abs(delta_temp)
        energy_noai = abs(self.temperature - 21.0)
        self.total_energy_ai   += energy_ai
        self.total_energy_noai += energy_noai
        reward = energy_noai - energy_ai
        if not (self.OPTIMAL_LOW <= self.temperature <= self.OPTIMAL_HIGH):
            penalty = abs(self.temperature - (self.OPTIMAL_LOW + self.OPTIMAL_HIGH) / 2) * 0.5
            reward -= penalty
        self.temperature = np.clip(self.temperature, self.MIN_TEMP, self.MAX_TEMP)
        return self._get_state(), reward

# ══════════════════════════════════════════════════════════════════════════════
# Agents
# ══════════════════════════════════════════════════════════════════════════════

class QLearningAgent:
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


class SARSAAgent(QLearningAgent):
    def update(self, state, action, reward, next_state, next_action=None):
        s  = self.discretize(state)
        ns = self.discretize(next_state)
        if next_action is None:
            next_action = self.select_action(next_state)
        next_q = self.q_table[ns][next_action]
        target = reward + self.gamma * next_q
        self.q_table[s][action] += self.alpha * (target - self.q_table[s][action])
        return next_action

# ══════════════════════════════════════════════════════════════════════════════
# Training & evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def train(agent, env_fn, cfg, early_stopping=False, progress_bar=None):
    rewards, energy_ai, energy_noai, epsilons = [], [], [], []
    best_reward, patience_ctr, best_q = -np.inf, 0, None
    stopped_at = None
    is_sarsa   = isinstance(agent, SARSAAgent)

    for ep in range(cfg['episodes']):
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
            else:
                patience_ctr += 1
            if patience_ctr >= cfg['early_stop_patience']:
                stopped_at = ep
                agent.q_table = best_q
                break

        agent.epsilon = max(cfg['epsilon_min'], agent.epsilon * cfg['epsilon_decay'])
        if progress_bar:
            progress_bar.progress((ep + 1) / cfg['episodes'])

    return rewards, energy_ai, energy_noai, epsilons, stopped_at


def evaluate_agent(agent, env_fn, runs=100):
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

# ══════════════════════════════════════════════════════════════════════════════
# Train button
# ══════════════════════════════════════════════════════════════════════════════

if not (run_qfull or run_qes or run_dq or run_sarsa):
    st.warning("Select at least one agent in the sidebar.")
    st.stop()

if st.button("🚀 Train Agents", type="primary", use_container_width=True):
    np.random.seed(int(seed))
    random.seed(int(seed))
    env_fn = lambda: DataCenterEnvironment(seed=None)

    agent_configs = []
    if run_qfull:  agent_configs.append(("Q-Learning Full",  False, False))
    if run_qes:    agent_configs.append(("Q-Learning ES",    False, True))
    if run_dq:     agent_configs.append(("Double Q",         True,  False))
    if run_sarsa:  agent_configs.append(("SARSA",            False, False))

    training_data = {}
    trained_agents = {}

    for name, use_double, use_es in agent_configs:
        st.markdown(f"**Training: {name}**")
        bar = st.progress(0)

        if name == "SARSA":
            agent = SARSAAgent(state_bins=state_bins, actions=5,
                               alpha=alpha, gamma=gamma,
                               epsilon=epsilon_start, seed=int(seed))
        else:
            agent = QLearningAgent(state_bins=state_bins, actions=5,
                                   alpha=alpha, gamma=gamma,
                                   epsilon=epsilon_start,
                                   double=use_double, seed=int(seed))

        r, ea, en, eps, stopped = train(agent, env_fn, CFG,
                                        early_stopping=use_es, progress_bar=bar)
        if stopped:
            st.caption(f"  ⏹ Early stopping at episode {stopped}")

        training_data[name] = dict(r=r, ea=ea, en=en, eps=eps)
        trained_agents[name] = agent

    # ── Evaluation ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Evaluation Results")

    results = {}
    for name, ag in trained_agents.items():
        ai, noai, trajs = evaluate_agent(ag, env_fn, runs=eval_runs)
        saving = 100 * (noai - ai) / noai
        results[name] = dict(ai=ai, noai=noai, trajs=trajs, saving=saving)

    # Summary table
    cols = st.columns(len(results))
    palette = ['#4C72B0','#DD8452','#55A868','#C44E52']
    for col, (name, r), color in zip(cols, results.items(), palette):
        col.metric(
            label=name,
            value=f"{r['saving'].mean():.1f}%",
            delta=f"±{r['saving'].std():.1f}% energy saved",
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    window = 20

    # Reward curves
    st.subheader("📈 Reward Curves")
    fig, ax = plt.subplots(figsize=(13, 4))
    styles = ['-','--','-.', ':']
    for (name, td), style in zip(training_data.items(), styles):
        r = td['r']
        ax.plot(r, alpha=0.2, linestyle=style, color='grey')
        ma = np.convolve(r, np.ones(window)/window, mode='valid')
        ax.plot(ma, label=name, linestyle=style, linewidth=2)
    ax.set_title("Reward per Episode (moving avg)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Energy savings bar chart
    st.subheader("⚡ Energy Savings Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    names  = list(results.keys())
    means  = [results[n]['saving'].mean() for n in names]
    stds   = [results[n]['saving'].std()  for n in names]
    colors_bar = palette[:len(names)]
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=colors_bar, alpha=0.85, edgecolor='k')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{m:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax.set_title(f"Mean Energy Saved vs. Rule-Based ({eval_runs} eval runs)")
    ax.set_ylabel("Energy Saved (%)")
    ax.set_ylim(0, max(means) * 1.35)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Temperature trajectories
    st.subheader("🌡️ Temperature Trajectory during Evaluation")
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec','Jan']
    n_agents = len(results)
    cols_per_row = 2
    rows = (n_agents + 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row,
                             figsize=(13, 4 * rows), sharey=True)
    axes_flat = axes.flat if n_agents > 1 else [axes]
    for ax, (name, r) in zip(axes_flat, results.items()):
        trajs  = r['trajs']
        mean_t = trajs.mean(axis=0)
        std_t  = trajs.std(axis=0)
        ax.fill_between(range(13), mean_t - std_t, mean_t + std_t,
                        alpha=0.25, label='±1 std')
        ax.plot(mean_t, linewidth=2, label='Mean temp')
        ax.axhspan(18, 24, alpha=0.12, color='green', label='Optimal band')
        ax.axhline(18, color='green', linestyle='--', linewidth=0.8)
        ax.axhline(24, color='green', linestyle='--', linewidth=0.8)
        ax.set_title(name)
        ax.set_xticks(range(13)); ax.set_xticklabels(months, fontsize=8)
        ax.set_ylabel("Temperature (°C)"); ax.set_xlabel("Month")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # hide unused axes
    for ax in list(axes_flat)[n_agents:]:
        ax.set_visible(False)
    plt.suptitle("Server Temperature Trajectory during Evaluation",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Q-table heatmap (best / Double-Q if available)
    heatmap_name = "Double Q" if "Double Q" in trained_agents else list(trained_agents.keys())[0]
    heatmap_agent = trained_agents[heatmap_name]
    SB = state_bins
    q_temp = heatmap_agent.q_table.mean(axis=(1, 2))

    st.subheader(f"🗺️ Q-table Heatmap — {heatmap_name}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    im = axes[0].imshow(q_temp, aspect='auto', cmap='RdYlGn', origin='lower')
    axes[0].set_title("Avg Q-values (over users/data bins)")
    axes[0].set_xlabel("Action  [−3  −1.5  0  +1.5  +3 °C]")
    axes[0].set_ylabel("Temperature bin  (0=cold  9=hot)")
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(['-3','-1.5','0','+1.5','+3'])
    plt.colorbar(im, ax=axes[0], label='Q-value')

    greedy        = np.argmax(q_temp, axis=1)
    action_labels = ['-3','-1.5','0','+1.5','+3']
    colors_act    = ['#1f77b4','#aec7e8','#98df8a','#ffbb78','#d62728']
    axes[1].barh(range(SB), [1]*SB,
                 color=[colors_act[a] for a in greedy], edgecolor='k')
    axes[1].set_title("Greedy Action per Temperature Bin")
    axes[1].set_xlabel("Preferred action"); axes[1].set_ylabel("Temperature bin")
    axes[1].set_yticks(range(SB))
    axes[1].set_yticklabels([f"bin {i}" for i in range(SB)])
    legend_handles = [mpatches.Patch(color=colors_act[i], label=action_labels[i])
                      for i in range(5)]
    axes[1].legend(handles=legend_handles, title="Action (°C)",
                   loc='lower right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Cumulative savings over training
    st.subheader("📉 Cumulative Energy Savings over Training")
    fig, ax = plt.subplots(figsize=(12, 4))
    for (name, td), style in zip(training_data.items(), styles):
        saved_pct = [(n - a) / n * 100 if n > 0 else 0
                     for a, n in zip(td['ea'], td['en'])]
        ax.plot(saved_pct, label=name, linestyle=style, linewidth=2)
    ax.set_title("Cumulative Energy Savings (%) over Training Episodes")
    ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Savings (%)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Download best model ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Download Best Model")

    best_name  = max(results, key=lambda n: results[n]['saving'].mean())
    best_agent = trained_agents[best_name]
    buf        = io.BytesIO()
    pickle.dump(best_agent.q_table, buf)
    buf.seek(0)

    st.success(
        f"Best agent: **{best_name}** — "
        f"mean saving {results[best_name]['saving'].mean():.2f}%"
    )
    st.download_button(
        label="⬇️ Download best_q_table.pkl",
        data=buf,
        file_name="best_q_table.pkl",
        mime="application/octet-stream",
    )

else:
    st.info("Configure hyperparameters in the sidebar and click **🚀 Train Agents** to begin.")

    # ── Environment explainer ──────────────────────────────────────────────────
    with st.expander("ℹ️ About the environment"):
        st.markdown("""
**State**: `[temperature, users, data_rate]`

**Actions**: 5 discrete cooling adjustments — −3 · −1.5 · 0 · +1.5 · +3 °C

**Reward**: energy saved vs. a rule-based baseline controller,
minus a penalty when server temperature leaves the optimal band (18–24 °C)

**Physics**:
- Server load (users + data rate) heats the server
- Atmospheric temperature drifts the server temperature naturally
- Monthly atmospheric temperatures cycle through a realistic annual pattern
        """)

    with st.expander("ℹ️ About the agents"):
        st.markdown("""
| Agent | Type | Key feature |
|---|---|---|
| **Q-Learning (full)** | Off-policy TD | Trains for all episodes |
| **Q-Learning (early stop)** | Off-policy TD | Stops when no improvement |
| **Double Q-Learning** | Off-policy TD | Two Q-tables to reduce overestimation |
| **SARSA** | On-policy TD | Updates use the actual next action taken |
        """)
