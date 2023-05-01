import gym
import time
import sys
import keyboard

env = gym.make("CartPole-v0")
env.reset()

TIMEOUT = 0.1

while True:
    done = False
    env.reset()
    while not done:
        env.render()

        # if space is pressed action 1 else 0
        if keyboard.is_pressed("space"):
            action = 1
        else:
            action = 0

        _, _, done, _ = env.step(action)
        # time.sleep(TIMEOUT)
