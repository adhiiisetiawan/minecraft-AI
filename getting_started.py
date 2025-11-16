import gym
import minerl
import logging
import os

# Set logging level (use INFO for less verbose output, DEBUG for more)
logging.basicConfig(level=logging.INFO)

# To see the UI, make sure you're NOT using xvfb-run
# If you want headless mode, use: xvfb-run -a python getting_started.py
# If you want to see the UI, run directly: python getting_started.py

print("Creating MineRL environment...")
print("Note: This will launch Minecraft, which takes time. Be patient!")
env = gym.make('MineRLBasaltFindCave-v0')

obs = env.reset()
done = False
step_count = 0

print("Environment ready! Starting gameplay...")
print("To see the Minecraft window, make sure you're running without xvfb-run")

while not done:
    # Take a random action
    action = env.action_space.sample()
    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    action["ESC"] = 0
    obs, reward, done, _ = env.step(action)
    
    # Render the environment (this will show the Minecraft window if not using xvfb)
    env.render()
    
    step_count += 1
    if step_count % 100 == 0:
        print(f"Step {step_count}, Reward: {reward}")

print(f"Episode finished after {step_count} steps")
env.close()