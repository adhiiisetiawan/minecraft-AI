import gym
import minerl
from tqdm import tqdm
import cv2
import numpy as np
import os
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)  # Set to INFO for less clutter


def get_action_sequence(repeats = 10):
    """
    Specify the action sequence for the agent to execute.
    Let's run around with sharp turns
    """
    action_sequence = []
    for _ in range(repeats):
      action_sequence += [''] * 20 # wait 1 sec
      action_sequence += ['forward attack jump'] * 20 # run forward + attack 1 sec
      action_sequence += ['camera:[0,-5]'] * 10 # turn 50deg
      action_sequence += ['forward attack jump'] * 20 # run forward + attack 1 sec
      action_sequence += ['camera:[0,10]'] * 10 # turn back 100deg
      action_sequence += ['forward attack jump'] * 40 # run forward + attack 2 secs
      action_sequence += ['camera:[0,-10]'] * 10 # turn 100deg
      action_sequence += ['forward attack jump'] * 20 # run forward + attack 1 secs
      action_sequence += ['camera:[0,5]'] * 10 # turn back 50deg
    return action_sequence

def str_to_act(env, actions):
    """
    Simplifies specifying actions for the scripted part of the agent.
    Some examples for a string with a single action:
        'craft:planks'
        'camera:[10,0]'
        'attack'
        'jump'
        ''
    There should be no spaces in single actions, as we use spaces to separate actions with multiple "buttons" pressed:
        'attack sprint forward'
        'forward camera:[0,10]'

    :param env: base MineRL environment.
    :param actions: string of actions.
    :return: dict action, compatible with the base MineRL environment.
    """
    act = env.action_space.noop()
    for action in actions.split():
        if ":" in action:
            k, v = action.split(':')
            if k == 'camera':
                act[k] = eval(v)
            else:
                act[k] = v
        else:
            act[action] = 1
    return act


def record_video_from_observations(observations, output_path, fps=20):
    """
    Record a video from a list of observation frames.
    
    :param observations: List of observation frames (numpy arrays)
    :param output_path: Path to save the video file
    :param fps: Frames per second for the output video
    """
    if not observations:
        print("No observations to record!")
        return
    
    # Get frame dimensions from first observation
    height, width = observations[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in observations:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to {output_path}")


# Create output directory for videos
video_dir = './videos'
os.makedirs(video_dir, exist_ok=True)

print("Creating MineRL environment...")
print("Note: This will launch Minecraft, which takes time. Be patient!")
env = gym.make('MineRLBasaltFindCave-v0')

action_sequence = get_action_sequence(repeats=5)
env.seed(2413)
obs = env.reset()

# Store observations for video recording
observations = []

# Initialize video writer if we want to record
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = os.path.join(video_dir, f'sequence_action_{timestamp}.mp4')

# Get frame dimensions from first observation
if 'pov' in obs:
    height, width = obs['pov'].shape[:2]
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    print(f"Recording video to {video_path}")
else:
    video_writer = None
    print("Warning: No 'pov' observation found, video recording disabled")

total_reward = 0
print("Environment ready! Starting action sequence...")
print("The Minecraft window should be visible (if not using xvfb-run)")

for i, action in enumerate(tqdm(action_sequence, desc="Executing actions")):
    obs, reward, done, _ = env.step(str_to_act(env, action))
    total_reward += reward
    
    # Render the environment (shows the Minecraft window)
    env.render()
    
    # Record frame if 'pov' observation is available
    if video_writer is not None and 'pov' in obs:
        frame = obs['pov']
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    if done:
        print(f"Episode finished at step {i}")
        break

# Release video writer
if video_writer is not None:
    video_writer.release()
    print(f"Video saved to {video_path}")

env.close()
print(f'\nTotal reward = {total_reward}')