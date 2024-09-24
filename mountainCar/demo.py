import gym
import time

environment_name = "MountainCar-v0"
env = gym.make(environment_name, render_mode='human')

num_steps = 1500

try:
    obs, info = env.reset()
    for step in range(num_steps):
        position, velocity = obs
        # If velocity is negative go left otherwise go right
        action = 0 if velocity < 0 else 2
        # Apply action to the environment
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.01)
        if done or truncated:
            obs, info = env.reset()
except Exception as e:
    print("An error occurred:", e)
finally:
    input("Press Enter to close the environment...")
    env.close()