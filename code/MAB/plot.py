import matplotlib.pyplot as plt
from agents import Agent


def plot_regret(agent_list: list[Agent], *, yscale: str = "lin") -> None:

    for agent in agent_list:
        if agent.turns == 0:
            print(f"Agent {agent.__class__.__name__} has not played yet.")
            continue

        plt.plot(agent.regret, label=agent.__class__.__name__)

    if yscale.lower() == "log":
        plt.yscale("log")
        plt.xscale("log")

    plt.xlabel("Turns")
    plt.ylabel("Regret")

    plt.legend()
    plt.show()
