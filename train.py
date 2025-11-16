import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import minerl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="MineRLBasaltFindCave-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-minerl",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--render", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, render the environment")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments (MineRL typically uses 1)")
    parser.add_argument("--num-steps", type=int, default=512,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, save model checkpoints")
    parser.add_argument("--save-interval", type=int, default=100,
        help="save model every N updates")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
        help="directory to save model checkpoints")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(gym_id, seed, idx, capture_video, run_name, render):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env, render

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNEncoder(nn.Module):
    """CNN encoder for processing image observations (pov)"""
    def __init__(self, input_shape):
        super(CNNEncoder, self).__init__()
        # Input shape: (H, W, 3) -> (3, H, W) for conv layers
        h, w, c = input_shape
        
        self.cnn = nn.Sequential(
            # First conv layer
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            # Second conv layer
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            # Third conv layer
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            dummy_output = self.cnn(dummy_input)
            self.feature_size = int(np.prod(dummy_output.shape[1:]))
    
    def forward(self, x):
        # x shape: (batch, H, W, 3) -> (batch, 3, H, W)
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)
        elif len(x.shape) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        
        x = x.float() / 255.0  # Normalize to [0, 1]
        features = self.cnn(x)
        return features.flatten(1)  # Flatten to (batch, features)


class Agent(nn.Module):
    """Hybrid PPO agent for MineRL with discrete buttons and continuous camera"""
    def __init__(self, env, device):
        super(Agent, self).__init__()
        self.device = device
        
        # Extract observation shape (pov image)
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            pov_shape = obs_space['pov'].shape  # (H, W, 3)
        else:
            pov_shape = obs_space.shape
        
        # CNN encoder for image observations
        self.cnn_encoder = CNNEncoder(pov_shape).to(device)
        feature_size = self.cnn_encoder.feature_size
        
        # Get action space info
        action_space = env.action_space
        if isinstance(action_space, gym.spaces.Dict):
            # Extract button actions (discrete) and camera (continuous)
            button_keys = [k for k in action_space.spaces.keys() 
                          if k != 'camera' and isinstance(action_space.spaces[k], gym.spaces.Discrete)]
            self.button_keys = button_keys
            self.num_button_actions = 2 ** len(button_keys)  # All combinations
            self.camera_dim = 2  # pitch, yaw
        else:
            raise ValueError("MineRL action space must be Dict")
        
        # Value function (critic)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(feature_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        
        # Discrete action head (buttons)
        self.actor_discrete = nn.Sequential(
            layer_init(nn.Linear(feature_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, self.num_button_actions), std=0.01),
        )
        
        # Continuous action head (camera)
        self.actor_camera_mean = nn.Sequential(
            layer_init(nn.Linear(feature_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, self.camera_dim), std=0.01),
        )
        self.actor_camera_logstd = nn.Parameter(torch.zeros(1, self.camera_dim))
    
    def get_value(self, x):
        """Get value estimate from observation"""
        features = self.cnn_encoder(x)
        return self.critic(features)
    
    def get_action_and_value(self, x, action_discrete=None, action_camera=None, compute_entropy=True):
        """Get action and value from observation
        
        Args:
            x: Observation tensor
            action_discrete: Pre-sampled discrete action (optional)
            action_camera: Pre-sampled camera action (optional)
            compute_entropy: Whether to compute entropy (skip for large action spaces to save memory)
        """
        features = self.cnn_encoder(x)
        
        # Discrete actions (buttons)
        discrete_logits = self.actor_discrete(features)
        discrete_dist = Categorical(logits=discrete_logits)
        if action_discrete is None:
            action_discrete = discrete_dist.sample()
        discrete_logprob = discrete_dist.log_prob(action_discrete)
        
        # Compute entropy only if requested (skip for large action spaces to save memory)
        # For very large action spaces (>100k), computing full entropy requires softmax over all actions
        # which can cause OOM even on large GPUs. Use approximation instead.
        if compute_entropy and self.num_button_actions < 100000:
            try:
                discrete_entropy = discrete_dist.entropy()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Fallback: use constant entropy estimate
                    compute_entropy = False
                else:
                    raise
        
        if not compute_entropy or self.num_button_actions >= 100000:
            # For large action spaces, use a constant entropy estimate to avoid OOM
            # This is a rough approximation: log(num_actions) scaled down
            # The entropy bonus encourages exploration, but for huge action spaces
            # we use a reasonable constant to avoid memory issues
            estimated_entropy = torch.tensor(np.log(max(2, min(self.num_button_actions, 10000))) * 0.3, 
                                           device=discrete_logprob.device, 
                                           dtype=discrete_logprob.dtype)
            discrete_entropy = estimated_entropy.expand_as(discrete_logprob)
        
        # Continuous actions (camera)
        camera_mean = self.actor_camera_mean(features)
        camera_logstd = self.actor_camera_logstd.expand_as(camera_mean)
        camera_std = torch.exp(camera_logstd)
        camera_dist = Normal(camera_mean, camera_std)
        if action_camera is None:
            action_camera = camera_dist.sample()
        camera_logprob = camera_dist.log_prob(action_camera).sum(1)
        camera_entropy = camera_dist.entropy().sum(1)
        
        # Combined entropy
        total_entropy = discrete_entropy + camera_entropy
        
        # Value
        value = self.critic(features)
        
        return (action_discrete, action_camera), (discrete_logprob, camera_logprob), total_entropy, value
    
    def action_to_dict(self, action_discrete, action_camera):
        """Convert discrete and continuous actions to MineRL action dict"""
        # Start with noop action - initialize all keys to 0
        action_dict = {}
        
        # Decode discrete action index to button states
        # Each bit in action_discrete represents one button
        action_idx = action_discrete.item() if isinstance(action_discrete, torch.Tensor) else action_discrete
        for i, key in enumerate(self.button_keys):
            action_dict[key] = int((action_idx >> i) & 1)
        
        # Add camera action (clip to reasonable range for pitch/yaw)
        if isinstance(action_camera, torch.Tensor):
            camera_vals = action_camera.cpu().numpy()
        else:
            camera_vals = np.array(action_camera)
        
        # Handle both 1D and 2D camera arrays
        if len(camera_vals.shape) == 1:
            camera_vals = camera_vals.reshape(1, -1)
        camera_vals = camera_vals[0]  # Take first if batched
        
        action_dict['camera'] = np.clip(camera_vals, -180, 180).tolist()
        
        # Ensure ESC is always 0 (prevents episode termination in BASALT environments)
        action_dict['ESC'] = 0
        
        return action_dict


def extract_pov(obs):
    """Extract pov image from observation dict and ensure contiguous array"""
    if isinstance(obs, dict):
        pov = obs['pov']
    else:
        pov = obs
    
    # Make a copy to ensure contiguous array (fixes negative stride issue)
    return np.ascontiguousarray(pov.copy())


if __name__ == "__main__":
    # Register signal handlers for cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        try:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Environment setup (MineRL doesn't support vectorized envs easily, so use single env)
    # Clean up any stale PID files from previous runs
    cleanup_minerl_processes()
    
    # Set environment variables to help with process watcher
    os.environ['MINERL_HEADLESS'] = '1'
    # Set temp directory for PID files if needed
    if 'TMPDIR' not in os.environ:
        os.environ['TMPDIR'] = '/tmp'
    
    print("Creating MineRL environment...")
    try:
        env = gym.make(args.gym_id)
    except Exception as e:
        # If environment creation fails, try cleaning up and retrying
        if "PID file" in str(e) or "process_watcher" in str(e).lower():
            print("Retrying after cleanup...")
            cleanup_minerl_processes()
            time.sleep(2)  # Wait a bit for processes to clean up
            env = gym.make(args.gym_id)
        else:
            raise
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if args.capture_video:
        # Create videos directory if it doesn't exist
        os.makedirs("videos", exist_ok=True)
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        print(f"Video recording enabled. Videos will be saved to videos/{run_name}/")
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    
    render_env = args.render
    
    # Create agent
    agent = Agent(env, device).to(device)
    
    # Print action space info for debugging
    print(f"Button keys: {agent.button_keys}")
    print(f"Number of button actions (2^{len(agent.button_keys)}): {agent.num_button_actions:,}")
    print(f"Camera dimension: {agent.camera_dim}")
    
    # Memory warning for large action spaces
    if agent.num_button_actions > 100000:
        print(f"\nWARNING: Very large action space ({agent.num_button_actions:,} actions)!")
        print("This may cause CUDA out of memory errors.")
        print("Entropy computation will use approximation to save memory.")
        print("Consider:")
        print("  1. Using CPU: --cuda False")
        print("  2. Reducing batch size: --num-steps 128 --num-minibatches 8")
        print("  3. Using a GPU with more memory")
        print()
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Create checkpoint directory
    if args.save_model:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Model checkpoints will be saved to {args.checkpoint_dir}/")
    
    # Track best reward for saving best model
    best_reward = float('-inf')
    
    # Storage setup
    obs_shape = env.observation_space['pov'].shape
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions_discrete = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    actions_camera = torch.zeros((args.num_steps, args.num_envs, 2)).to(device)
    logprobs_discrete = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_camera = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs_raw = env.reset()
    next_obs = torch.Tensor(extract_pov(next_obs_raw)).to(device).unsqueeze(0)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    
    print(f"Starting training for {num_updates} updates...")
    
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs.squeeze(0)
            dones[step] = next_done
            
            # Action logic
            with torch.no_grad():
                # During rollout, we don't need entropy (only need it for loss calculation during updates)
                # Skip entropy computation to save memory for large action spaces
                (action_discrete, action_camera), (logprob_discrete, logprob_camera), _, value = agent.get_action_and_value(
                    next_obs, 
                    compute_entropy=False  # Skip entropy during rollout to save memory
                )
                values[step] = value.flatten()
            
            actions_discrete[step] = action_discrete
            actions_camera[step] = action_camera
            logprobs_discrete[step] = logprob_discrete
            logprobs_camera[step] = logprob_camera
            
            # Convert to action dict
            action_dict = agent.action_to_dict(action_discrete, action_camera)
            
            # Execute game step
            next_obs_raw, reward, done, info = env.step(action_dict)
            if render_env:
                env.render()
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(extract_pov(next_obs_raw)).to(device).unsqueeze(0)
            next_done = torch.Tensor([done]).to(device)
            
            if "episode" in info.keys():
                episode_return = info["episode"]["r"]
                print(f"global_step={global_step}, episodic_return={episode_return}, episodic_length={info['episode']['l']}")
                writer.add_scalar("charts/episodic_return", episode_return, global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                
                # Save best model
                if args.save_model and episode_return > best_reward:
                    best_reward = episode_return
                    best_model_path = os.path.join(args.checkpoint_dir, f"best_model_{run_name}.pth")
                    torch.save({
                        'agent_state_dict': agent.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                        'episode_return': episode_return,
                        'args': vars(args),
                        'button_keys': agent.button_keys,
                        'num_button_actions': agent.num_button_actions,
                        'camera_dim': agent.camera_dim,
                    }, best_model_path)
                    print(f"Saved best model with reward {episode_return:.2f} to {best_model_path}")
        
        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
        
        # Flatten the batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs_discrete = logprobs_discrete.reshape(-1)
        b_logprobs_camera = logprobs_camera.reshape(-1)
        b_actions_discrete = actions_discrete.reshape(-1)
        b_actions_camera = actions_camera.reshape((-1, 2))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                # Skip entropy computation during updates for large action spaces (saves memory)
                # We'll use a simple approximation instead
                compute_entropy = agent.num_button_actions < 100000  # Only compute if action space is reasonable
                (_, _), (newlogprob_discrete, newlogprob_camera), entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    b_actions_discrete[mb_inds],
                    b_actions_camera[mb_inds],
                    compute_entropy=compute_entropy
                )
                
                # Combined log probability
                newlogprob = newlogprob_discrete + newlogprob_camera
                oldlogprob = b_logprobs_discrete[mb_inds] + b_logprobs_camera[mb_inds]
                
                logratio = newlogprob - oldlogprob
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Record metrics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        print(f"Update {update}/{num_updates}, SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)
        
        # Save periodic checkpoint
        if args.save_model and update % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{update}_{run_name}.pth")
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'update': update,
                'args': vars(args),
                'button_keys': agent.button_keys,
                'num_button_actions': agent.num_button_actions,
                'camera_dim': agent.camera_dim,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    if args.save_model:
        final_model_path = os.path.join(args.checkpoint_dir, f"final_model_{run_name}.pth")
        torch.save({
            'agent_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'args': vars(args),
            'button_keys': agent.button_keys,
            'num_button_actions': agent.num_button_actions,
            'camera_dim': agent.camera_dim,
        }, final_model_path)
        print(f"Saved final model to {final_model_path}")
    
    env.close()
    writer.close()
    print("Training completed!")

