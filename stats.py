import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


class Stats:
    def __init__(self, window: int, decay_type : str):
        self.window = window
        self.decay_type = decay_type
        self.rewards = []
        self.ep_len = []
        self.captured = []
        self.truncated = []
        self.summary = []

    def get_moving_aves(self, data):
        return np.convolve(np.array(data).flatten(), np.ones(self.window) / self.window, mode="same")

    def add_result(self, captured: bool, truncated: bool):
        self.captured.append(captured)
        self.truncated.append(truncated)

    def add_episode(self, reward: float, length: int):
        self.rewards.append(reward)
        self.ep_len.append(length)

    def add_summary(self, episode: int, ave_steps: float, ave_reward: float, success_rate: float, epsilon: float):
        self.summary.append({
            "episode": episode,
            "ave_steps": ave_steps,
            "ave_reward": ave_reward,
            "success_rate": success_rate,
            "epsilon": epsilon
        })

    def plot(self, cat_name: str, save_dir: str = "plots"):
        os.makedirs(save_dir, exist_ok=True)
        cat_name = cat_name.capitalize()
        episodes = np.arange(len(self.rewards))
        colors = ["green" if c else "red" for c in self.captured]

        fig = plt.figure(figsize=(14, 7.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[3.5, 1.5], wspace=0.1, hspace=0.3)

        ax_reward = fig.add_subplot(gs[0, 0])
        ax_len = fig.add_subplot(gs[1, 0])
        ax_summary = fig.add_subplot(gs[:, 1])

        fig.suptitle(f"CatBot vs. {cat_name}", fontsize=18, fontweight="bold")

        # Reward plot
        ax_reward.scatter(episodes, self.rewards, c=colors, s=7, alpha=0.18)
        reward_ave = self.get_moving_aves(self.rewards)
        ax_reward.plot(episodes, reward_ave, color="black", linewidth=1.8)

        ax_reward.set_title("Average Episode Reward", fontsize=10, fontweight="bold")
        ax_reward.set_xlabel("Episode", fontsize=8.5)
        ax_reward.set_ylabel("Reward per Episode", fontsize=8.5)
        ax_reward.grid(alpha=0.25)

        reward_handles = [
            Line2D([0], [0], color="black", linewidth=1.8, label=f"Reward ({self.window}-Ep ave)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=5, label="Caught"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=5, label="Truncated")
        ]
        ax_reward.legend(handles=reward_handles, loc="lower right", fontsize=7.5, framealpha=0.9)

        # Episode length plot
        ax_len.scatter(episodes, self.ep_len, c=colors, s=7, alpha=0.18)
        ave_len = self.get_moving_aves(self.ep_len)
        ax_len.plot(episodes, ave_len, color="#1f77b4", linestyle="--", linewidth=1.8)

        ax_len.set_title("Average Episode Length", fontsize=10, fontweight="bold")
        ax_len.set_xlabel("Episode", fontsize=8.5)
        ax_len.set_ylabel("Steps per Episode", fontsize=8.5)
        ax_len.grid(alpha=0.25)

        len_handles = [
            Line2D([0], [0], color="#1f77b4", linestyle="--", linewidth=1.8, label=f"Len ({self.window}-Ep ave)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=5, label="Caught"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=5, label="Truncated")
        ]
        ax_len.legend(handles=len_handles, loc="upper right", fontsize=7.5, framealpha=0.9)

        # Summary
        ax_summary.axis("off")
        total = len(self.rewards)
        cap_rate = sum(self.captured) / total * 100 if total else 0.0
        trunc_rate = sum(self.truncated) / total * 100 if total else 0.0
        cap_steps = [steps for steps, captured in zip(self.ep_len, self.captured) if captured]

        ave_reward = np.mean(self.rewards) if total else 0.0
        ave_ep_steps = np.mean(self.ep_len) if total else 0.0
        ave_cap_steps = np.mean(cap_steps) if cap_steps else 0.0

        summary_text = (
            "SUMMARY\n"
            f"Episodes:           {total:,}\n"
            f"Capture Rate:       {cap_rate:.1f}%\n"
            f"Truncation Rate:    {trunc_rate:.1f}%\n\n"
            f"Average Reward:     {ave_reward:.1f}\n"
            f"Avg Episode Steps:  {ave_ep_steps:.1f}\n"
            f"Avg Capture Steps:  {ave_cap_steps:.1f}\n"
            f"Decay Type:          {self.decay_type.capitalize()}\n"
        )

        if self.summary:
            summary_text += f"{'Ep':>6} {'Steps':>6} {'Reward':>7} {'Win%':>6} {'ε':>5}\n"
            summary_text += "-----------------------------------\n"
            for p in self.summary:
                summary_text += f"{p['episode']:>6} {p['ave_steps']:>6.1f} {p['ave_reward']:>7.1f} {p['success_rate']:>6.1%} {p['epsilon']:>5.2f}\n"

        ax_summary.text(
            0.05, 0.95, summary_text,
            fontsize=9.5, fontfamily="monospace", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#fafafa", edgecolor="#d0d0d0")
        )

        plt.savefig(os.path.join(save_dir, f"CatBotv{cat_name}{self.decay_type.upper()}_stats.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
