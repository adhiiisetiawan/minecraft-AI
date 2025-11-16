import argparse
import os
import torch
import gym
import minerl
import numpy as np
import cv2
from datetime import datetime
from distutils.util import strtobool
from train import Agent, extract_pov
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with trained PPO model")
    parser.add_argument("--checkpoint", type=str, required=True,
        help="Path to model checkpoint file (.pth)")
    parser.add_argument("--gym-id", type=str, default="MineRLBasaltFindCave-v0",
        help="The id of the gym environment")
    parser.add_argument("--seed", type=int, default=None,
        help="Random seed (uses checkpoint seed if not specified)")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use CUDA if available")
    parser.add_argument("--render", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Render the environment window")
    parser.add_argument("--record-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Record video of inference")
    parser.add_argument("--max-steps", type=int, default=10000,
        help="Maximum number of steps to run")
    parser.add_argument("--num-episodes", type=int, default=1,
        help="Number of episodes to run")
    parser.add_argument("--output-dir", type=str, default="inference_outputs",
        help="Directory to save videos and logs")
    
    args = parser.parse_args()
    return args


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract saved information
    saved_args = checkpoint.get('args', {})
    button_keys = checkpoint.get('button_keys', [])
    num_button_actions = checkpoint.get('num_button_actions', 0)
    camera_dim = checkpoint.get('camera_dim', 2)
    
    print(f"Model info:")
    print(f"  Button keys: {button_keys}")
    print(f"  Number of button actions: {num_button_actions:,}")
    print(f"  Camera dimension: {camera_dim}")
    if 'episode_return' in checkpoint:
        print(f"  Best episode return: {checkpoint['episode_return']:.2f}")
    if 'global_step' in checkpoint:
        print(f"  Training steps: {checkpoint['global_step']:,}")
    
    return checkpoint, saved_args, button_keys, num_button_actions, camera_dim


def create_agent_from_checkpoint(env, checkpoint, device, button_keys, num_button_actions, camera_dim):
    """Create agent and load weights from checkpoint"""
    # Create agent with same architecture
    agent = Agent(env, device).to(device)
    
    # Override action space info if available from checkpoint
    if button_keys:
        agent.button_keys = button_keys
        agent.num_button_actions = num_button_actions
        agent.camera_dim = camera_dim
    
    # Load state dict
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()  # Set to evaluation mode (no gradients)
    
    print("Model loaded successfully!")
    return agent


def record_video_frame(video_writer, obs):
    """Record a single frame to video"""
    if video_writer is not None and 'pov' in obs:
        frame = obs['pov']
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)


def run_inference(args):
    """Run inference with loaded model"""
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint, saved_args, button_keys, num_button_actions, camera_dim = load_model(args.checkpoint, device)
    
    # Create environment
    print(f"Creating environment: {args.gym_id}")
    env = gym.make(args.gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Set seed
    seed = args.seed if args.seed is not None else saved_args.get('seed', 1)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    # Create agent
    agent = create_agent_from_checkpoint(env, checkpoint, device, button_keys, num_button_actions, camera_dim)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run episodes
    for episode in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{args.num_episodes}")
        print(f"{'='*60}")
        
        # Setup video recording
        video_writer = None
        if args.record_video:
            obs_shape = env.observation_space['pov'].shape
            height, width = obs_shape[:2]
            fps = 20
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(args.output_dir, f'inference_ep{episode+1}_{timestamp}.mp4')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            print(f"Recording video to {video_path}")
        
        # Reset environment
        obs_raw = env.reset()
        obs = torch.Tensor(extract_pov(obs_raw)).to(device).unsqueeze(0)
        done = False
        step_count = 0
        total_reward = 0
        
        print("Starting inference...")
        
        while not done and step_count < args.max_steps:
            # Get action from policy (no gradients)
            with torch.no_grad():
                (action_discrete, action_camera), _, _, _ = agent.get_action_and_value(obs)
            
            # Convert to action dict
            action_dict = agent.action_to_dict(action_discrete, action_camera)
            
            # Step environment
            obs_raw, reward, done, info = env.step(action_dict)
            
            # Record frame
            record_video_frame(video_writer, obs_raw)
            
            # Render if requested
            if args.render:
                env.render()
            
            # Update observation
            obs = torch.Tensor(extract_pov(obs_raw)).to(device).unsqueeze(0)
            
            total_reward += reward
            step_count += 1
            
            # Print progress
            if step_count % 100 == 0:
                print(f"  Step {step_count}, Reward: {total_reward:.2f}")
        
        # Close video writer
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to {video_path}")
        
        # Print episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Done: {done}")
        if "episode" in info.keys():
            print(f"  Episode return: {info['episode']['r']:.2f}")
            print(f"  Episode length: {info['episode']['l']}")
    
    env.close()
    print(f"\n{'='*60}")
    print("Inference completed!")
    print(f"Outputs saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        exit(1)
    
    run_inference(args)

