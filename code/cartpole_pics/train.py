# %%
import stable_baselines3 as sb3
import gym
import time

# %%
env = gym.make("CartPole-v0")
agent = sb3.PPO("MlpPolicy", env, verbose=1)

# %%
agent.learn(100_000)

# %%
# play and render, pasuing when the episode is done
obs = env.reset()
done = False
while True:
    while not done:
        env.render()
        action, _ = agent.predict(obs)
        obs, _, done, _ = env.step(action)
        time.sleep(1 / 50)
    input("Press enter to continue")
    done = False

# %%
