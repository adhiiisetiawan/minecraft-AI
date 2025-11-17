# Minecraft AI

<div align="center">
  <img src="assets/output.gif" alt="Minecraft AI Training Demo" width="600" />
  <br />
  <em>🎮 AI Agent Learning to Navigate and Explore in Minecraft</em>
</div>

---

A project for experimenting with reinforcement learning in Minecraft using the MineRL environment. This project includes scripts for running random actions and executing predefined action sequences with video recording capabilities.

**Status**: On going development

## Overview

This project uses [MineRL](https://github.com/minerllabs/minerl), a research platform for reinforcement learning in Minecraft. It includes:

- **Random action execution**: Test the environment with random actions
- **Scripted action sequences**: Execute predefined action patterns
- **Video recording**: Record gameplay videos automatically
- **Visual rendering**: Display Minecraft window during execution

## Requirements

- **Python**: 3.11 (required - MineRL dependencies need Python 3.11 or earlier)
- **Conda**: For environment management
- **Java**: Required for Minecraft (usually comes with MineRL)
- **OpenCV**: For video recording (opencv-python)

## Installation

### Option 1: Using environment.yml (Recommended)

The easiest way to set up the environment is using the provided `environment.yml` file:

```bash
# Create and activate environment from YAML file
conda env create -f environment.yml
conda activate minecraft-ai
```

**Note**: If the environment already exists and you want to update it:
```bash
conda env update -f environment.yml --prune
```

This will install all dependencies including:
- Python 3.11.14
- numpy==1.23.5 (compatible with Python 3.11)
- gym==0.23.1
- minerl==1.0.2
- torch==2.9.1 and torchvision==0.24.1
- opencv-python==4.11.0.86
- tensorboard==2.20.0
- tqdm==4.67.1
- And all other required dependencies

### Option 2: Manual Installation

If you prefer to install manually:

```bash
# Step 1: Create a new conda environment with Python 3.11
conda create -n minecraft-ai python=3.11 -y

# Step 2: Activate the environment
conda activate minecraft-ai

# Step 3: Install dependencies
pip install gym==0.23.1
pip install minerl==1.0.2
pip install numpy==1.23.5  # Important: numpy 1.23.5 for Python 3.11 compatibility
pip install opencv-python
pip install tqdm
pip install torch==2.9.1 torchvision==0.24.1  # For PPO training
pip install tensorboard  # For training visualization
```

**Note**: If you encounter issues with numpy installation, you may need to install it with:
```bash
pip install numpy==1.23.5 --no-build-isolation
```

### Step 3: Verify Installation

```bash
python -c "import gym; import minerl; import torch; print('Installation successful!')"
```

## Docker Installation (Alternative)

Docker provides an isolated environment with all dependencies pre-configured, including Python 3.11, OpenJDK 8, and xvfb-run.

### Prerequisites

- Docker installed on your system
- For GPU support: NVIDIA GPU with drivers and NVIDIA Container Toolkit

### Step 1: Install NVIDIA Container Toolkit (For GPU Support)

If you want to use GPU acceleration, install the NVIDIA Container Toolkit:

```bash
# On Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Verify GPU access:**
```bash
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Step 2: Build Docker Image

```bash
# Build the image
sudo docker build -t minecraft-ai .
```

This will create an image (~12GB) with all dependencies including:
- Python 3.11
- OpenJDK 8
- xvfb-run
- CUDA 11.8 runtime (for GPU support)
- All Python packages from requirements.txt

### Step 3: Run Container

#### CPU Mode (No GPU)

```bash
# Interactive shell
sudo docker run -it --rm \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/logs:/app/logs \
  minecraft-ai

# Run training script
sudo docker run -it --rm \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/logs:/app/logs \
  minecraft-ai \
  xvfb-run -a python train.py --total-timesteps 1000000 --cuda False
```

#### GPU Mode (With CUDA)

```bash
# Interactive shell with GPU
sudo docker run -it --rm --gpus all \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/logs:/app/logs \
  minecraft-ai

# Run training with GPU
sudo docker run -it --rm --gpus all \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/logs:/app/logs \
  minecraft-ai \
  xvfb-run -a python train.py --total-timesteps 1000000 --cuda True

# Verify CUDA in container
sudo docker run -it --rm --gpus all minecraft-ai \
  python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Using Docker Compose

Easier way to manage containers:

```bash
# Build and run (with GPU support enabled)
sudo docker-compose up

# Run in background
sudo docker-compose up -d

# View logs
sudo docker-compose logs -f

# Stop
sudo docker-compose down
```

**Note**: The `docker-compose.yml` file is configured for GPU support. If you don't have a GPU, edit the file and remove or comment out the `deploy` section.

### Docker Tips

- **Volume Mounts**: The `-v` flags mount local directories so outputs persist outside the container
- **GPU Access**: The `--gpus all` flag is required for CUDA support
- **Permissions**: You may need `sudo` depending on your Docker setup
- **Rebuild**: If you change dependencies, rebuild with `sudo docker build -t minecraft-ai .`

## Project Structure

```
minecraft-AI/
├── README.md              # This file
├── Dockerfile             # Docker image definition (Python 3.11, OpenJDK 8, xvfb, CUDA)
├── docker-compose.yml     # Docker Compose configuration
├── environment.yml        # Conda environment file with all dependencies
├── requirements.txt       # Python package requirements
├── getting_started.py     # Random action execution script
├── sequence_action.py     # Scripted action sequence with video recording
├── train.py              # PPO implementation for MineRL (hybrid action space)
├── inference.py           # Inference script for running trained models
├── checkpoints/           # Saved model checkpoints (created during training)
├── inference_outputs/     # Inference videos and outputs (created during inference)
├── videos/                # Output directory for recorded videos
├── runs/                  # TensorBoard logs
└── logs/                  # Minecraft and environment logs
```

## Usage

### 1. Random Actions (`getting_started.py`)

Run the environment with random actions:

```bash
# With visual window (requires display)
python getting_started.py

# Headless mode (no window, for servers)
xvfb-run -a python getting_started.py
```

**What it does:**
- Creates a MineRL environment (`MineRLBasaltFindCave-v0`)
- Executes random actions
- Displays the Minecraft window (if not using xvfb-run)
- Prints progress every 100 steps

### 2. Scripted Action Sequence (`sequence_action.py`)

Execute a predefined action sequence with video recording:

```bash
# With visual window
python sequence_action.py

# Headless mode
xvfb-run -a python sequence_action.py
```

**What it does:**
- Executes a scripted sequence of actions (forward, attack, jump, camera movements)
- Records video automatically to `./videos/` directory
- Shows progress with tqdm
- Saves video with timestamp

**Action Sequence Pattern:**
- Wait → Run forward + attack + jump → Turn camera → Repeat
- The pattern repeats 5 times by default
- You can modify `repeats` parameter in `get_action_sequence()` function

### 3. PPO Training (`train.py`)

Train a PPO agent to learn to find caves in Minecraft:

```bash
# Basic training (headless, CPU - recommended for large action spaces)
xvfb-run -a python train.py --total-timesteps 1000000 --cuda False

# Basic training with GPU (if you have 12GB+ GPU memory)
xvfb-run -a python train.py --total-timesteps 1000000

# With rendering (see agent learn)
python train.py --total-timesteps 1000000 --render --cuda False

# With video capture (saves videos automatically)
python train.py --total-timesteps 1000000 --capture-video --cuda False

# Memory-efficient training (smaller batches)
python train.py --total-timesteps 1000000 --num-steps 128 --num-minibatches 8 --cuda False

# With Weights & Biases tracking
python train.py --total-timesteps 1000000 --track --wandb-project-name my-project --cuda False
```

**What it does:**
- Trains a PPO agent using Proximal Policy Optimization
- Uses CNN to process image observations
- Handles hybrid action space (discrete buttons + continuous camera)
- Logs training metrics to TensorBoard (in `runs/` directory)
- Saves videos of agent performance automatically (if `--capture-video` is enabled)
- Prints action space information and memory warnings at startup

**Key Features:**
- **CNN Encoder**: 3-layer convolutional network for image feature extraction
- **Hybrid Action Space**: Handles both discrete button actions and continuous camera control
- **MineRL-Adapted Hyperparameters**: Optimized for MineRL environment characteristics
- **Memory Diagnostics**: Automatically detects and warns about large action spaces
- **Video Recording**: Built-in video capture support for monitoring training progress
- **Model Saving**: Automatically saves checkpoints and best models during training

**Model Saving:**
- **Best Model**: Saved automatically when a new best reward is achieved
- **Periodic Checkpoints**: Saved every N updates (default: every 100 updates)
- **Final Model**: Saved at the end of training
- All models saved to `checkpoints/` directory with metadata

### 4. Model Inference (`inference.py`)

Run inference with a trained model:

```bash
# Basic inference (with video recording)
python inference.py --checkpoint checkpoints/best_model_*.pth

# Run multiple episodes
python inference.py --checkpoint checkpoints/best_model_*.pth --num-episodes 5

# With rendering (see the agent play)
python inference.py --checkpoint checkpoints/best_model_*.pth --render

# Headless mode (no window)
xvfb-run -a python inference.py --checkpoint checkpoints/best_model_*.pth

# Custom output directory
python inference.py --checkpoint checkpoints/best_model_*.pth --output-dir my_inference_results
```

**What it does:**
- Loads a trained model from checkpoint
- Runs the agent in the environment
- Records video automatically (saved to `inference_outputs/`)
- Shows agent performance without training
- Supports multiple episodes

**Checkpoint Files:**
- `best_model_*.pth`: Best performing model (highest reward)
- `checkpoint_N_*.pth`: Periodic checkpoints during training
- `final_model_*.pth`: Final model at end of training

## Environment Details

### Action Space

MineRL uses a **hybrid action space** (Dict space):

- **Discrete/Binary actions**: Button presses (forward, attack, jump, etc.)
- **Continuous actions**: Camera rotation (pitch, yaw)

### Observation Space

The environment provides:
- `pov`: First-person view image (RGB array)
- Other state information (inventory, position, etc.)

### Reward Structure

`MineRLBasaltFindCave-v0` is a BASALT environment with:
- **Sparse rewards**: Rewards given when finding caves
- **Episode termination**: Episode ends when cave is found or time limit reached

## Display Setup (WSL2/Linux)

If you're running on WSL2 or a Linux system without a display, you have two options:

### Option 1: Use xvfb (Headless)

```bash
# Install xvfb
sudo apt-get install xvfb

# Run scripts with xvfb
xvfb-run -a python getting_started.py
```

### Option 2: X11 Forwarding (See Window)

1. Install an X server on Windows (VcXsrv, X410, or WSLg)
2. Set DISPLAY variable:
   ```bash
   export DISPLAY=:0
   # or for VcXsrv:
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
   ```
3. Run scripts normally:
   ```bash
   python getting_started.py
   ```

## Troubleshooting

### Issue: NumPy installation fails

**Solution**: Use Python 3.11 and install numpy 1.23.5:
```bash
conda activate minecraft-ai
pip install numpy==1.23.5 --no-build-isolation
```

### Issue: ModuleNotFoundError: No module named 'distutils'

**Solution**: This happens with Python 3.12+. Use Python 3.11:
```bash
conda create -n minecraft-ai python=3.11 -y
```

### Issue: Gym deprecation warnings

**Note**: Gym is deprecated in favor of Gymnasium. The warnings are harmless but you can suppress them by setting logging level to ERROR in the scripts.

### Issue: Minecraft window doesn't appear

**Solutions**:
1. Make sure you're not using `xvfb-run` if you want to see the window
2. Check DISPLAY variable is set correctly (for WSL2)
3. Verify X server is running (for WSL2)

### Issue: Video recording not working

**Solution**: Ensure opencv-python is installed:
```bash
pip install opencv-python
```

### Issue: Process watcher PID file error

**Problem**: `Unable to create/open PID file` or `subprocess.CalledProcessError` when creating MineRL environment.

**Solutions**:

1. **Clean up stale PID files** (automatic in train.py):
   ```bash
   # Manual cleanup if needed
   rm -f /tmp/minerl_watcher_*.pid ./minerl_watcher_*.pid
   ```

2. **Kill any stuck MineRL processes**:
   ```bash
   pkill -f "minerl.utils.process_watcher"
   pkill -f "launchClient.sh"
   ```

3. **Check file permissions**:
   ```bash
   # Ensure /tmp is writable
   ls -ld /tmp
   chmod 1777 /tmp  # If needed (requires sudo)
   ```

4. **Run with proper environment**:
   ```bash
   # Set TMPDIR if needed
   export TMPDIR=/tmp
   python train.py
   ```

5. **In Docker, ensure proper permissions**:
   ```bash
   # Run with proper user or use tmpfs
   docker run -it --rm --tmpfs /tmp minecraft-ai
   ```

The code now automatically cleans up stale PID files on startup and retries if environment creation fails.

### Issue: CUDA out of memory (OOM) error

**Problem**: MineRL has many button actions, creating a large discrete action space (2^n combinations). This can require 8-12GB+ GPU memory.

**Solutions** (in order of recommendation):

1. **Use CPU instead of GPU** (easiest):
   ```bash
   python train.py --cuda False --total-timesteps 1000000
   ```
   - Slower but avoids GPU memory limits
   - CPU RAM is usually larger (16GB+ common)

2. **Reduce batch size**:
   ```bash
   python train.py --num-steps 128 --num-minibatches 8 --total-timesteps 1000000
   ```
   - Smaller batches = less memory per operation
   - Trade-off: slightly less stable gradients

3. **Use a GPU with more memory**:
   - 12GB+ GPU recommended for large action spaces
   - Or use cloud GPU services (AWS, GCP, etc.)

4. **Check action space size**:
   - The code now prints action space info at startup
   - Look for: "Number of button actions: X"
   - If X > 100,000, consider using CPU or reducing batch size

**Memory Requirements:**
- **Small action space** (< 10,000 actions): ~2-4 GB GPU memory
- **Medium action space** (10,000 - 100,000): ~4-8 GB GPU memory  
- **Large action space** (> 100,000 actions): ~8-12+ GB GPU memory or use CPU

### Issue: Negative stride error when converting observations

**Error**: `ValueError: At least one stride in the given numpy array is negative`

**Solution**: This has been fixed in the latest version. The `extract_pov()` function now ensures contiguous arrays. If you still see this error, update to the latest code.

### Issue: Action space too large warning

**Warning**: "WARNING: Very large action space (X actions)!"

**Explanation**: MineRL has many button actions (forward, back, attack, jump, etc.). When treating all combinations as a discrete space, this creates 2^n possible actions where n = number of buttons.

**What to do**:
- The warning is informational - training will still work
- Use CPU (`--cuda False`) if you see CUDA OOM errors
- Consider reducing batch size if memory is tight

### Issue: CUDA not detected in Docker container

**Problem**: `torch.cuda.is_available()` returns `False` in Docker container.

**Solutions**:

1. **Install NVIDIA Container Toolkit** (if not installed):
   ```bash
   # Follow Step 1 in Docker Installation section
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Use `--gpus all` flag** when running container:
   ```bash
   # Wrong (no GPU access)
   sudo docker run -it minecraft-ai
   
   # Correct (with GPU access)
   sudo docker run -it --gpus all minecraft-ai
   ```

3. **Verify GPU is accessible**:
   ```bash
   # Test on host
   nvidia-smi
   
   # Test in container
   sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

4. **Check Docker Compose GPU configuration**:
   - Ensure `docker-compose.yml` has the `deploy.resources.reservations.devices` section enabled
   - For CPU-only, remove or comment out the GPU section

5. **Rebuild image** if you changed Dockerfile:
   ```bash
   sudo docker build -t minecraft-ai .
   ```

## PPO Implementation Details

### Architecture Overview

The PPO implementation (`train.py`) uses a hybrid architecture to handle MineRL's unique characteristics:

#### 1. CNN Encoder for Image Observations

**Why CNN?**
- MineRL provides observations as first-person view images (`pov`)
- Images contain rich spatial information (terrain, structures, caves)
- CNNs excel at extracting hierarchical features from images

**Architecture Choice:**
- **3 Convolutional Layers**: Sufficient depth to capture spatial patterns without over-parameterization
  - Layer 1: 32 filters, 8x8 kernel, stride 4 (coarse features)
  - Layer 2: 64 filters, 4x4 kernel, stride 2 (mid-level features)
  - Layer 3: 64 filters, 3x3 kernel, stride 1 (fine-grained features)
- **ReLU Activations**: Standard for CNNs, helps with gradient flow
- **Feature Vector**: Flattened CNN output feeds into actor/critic networks

**Why 3+ Layers?**
- Single layer: Too shallow, misses complex patterns
- 2 layers: Better but limited feature hierarchy
- 3+ layers: Captures multi-scale features (edges → textures → objects → scenes)
- More layers: Diminishing returns, risk of overfitting with limited data

#### 2. Hybrid Action Space Handling

**Why Hybrid?**
MineRL's action space combines:
- **Discrete buttons**: Binary actions (forward, back, attack, jump, etc.)
- **Continuous camera**: Precise rotation control (pitch, yaw)

**Implementation Approach:**
- **Discrete Actions as Combined Space**: All button combinations treated as single discrete action
  - Each combination encoded as integer (bit representation)
  - Categorical distribution over all possible combinations
  - Enables learning which button combinations are effective together
  - **Note**: Action space size = 2^(number of buttons), which can be very large
    - Example: 20 buttons = 1,048,576 possible actions
    - The code automatically prints action space size and warns if > 100,000 actions
  
- **Continuous Camera Control**: Separate head with Normal distribution
  - Mean predicted by network
  - Learnable log_std parameter for exploration
  - Clipped to reasonable range (-180 to 180 degrees)

**Why Combined Discrete Space?**
- Treating buttons independently would require multi-discrete action space
- Combined approach learns correlations between button presses
- More efficient: single discrete head vs. multiple binary heads
- Matches how humans play: coordinated button presses

**Memory Considerations:**
- Large action spaces require significant GPU memory for softmax operations
- The code includes automatic diagnostics and warnings
- CPU training is recommended for very large action spaces (> 100,000 actions)
- Batch size can be reduced to save memory if needed

#### 3. Hyperparameter Choices

**Environment-Specific Adaptations:**

- **`num_envs: 1`**: Single environment
  - MineRL environments are computationally expensive (full Minecraft instance)
  - Vectorized environments difficult to implement (Java process overhead)
  - Single env allows focus on sample efficiency vs. parallelization

- **`num_steps: 512`**: Shorter rollouts
  - MineRL episodes can be very long (finding caves takes time)
  - Shorter rollouts provide more frequent updates
  - Balances exploration (longer) vs. learning speed (shorter)
  - Prevents stale data in long episodes

- **`learning_rate: 3e-4`**: Standard PPO learning rate
  - Works well for image-based RL
  - Not too aggressive (avoids instability)
  - Not too conservative (allows learning in reasonable time)

- **`total_timesteps: 1e6`**: Extended training
  - MineRL tasks are complex (sparse rewards, large state space)
  - Image observations require more samples to learn useful features
  - BASALT tasks typically need millions of steps

- **`ent_coef: 0.01`**: Moderate exploration
  - Encourages exploration in large action space
  - Prevents premature convergence to suboptimal policies
  - Balances exploration vs. exploitation

- **`update_epochs: 4`**: Standard PPO updates
  - Multiple passes over collected data
  - Efficient use of samples
  - Prevents overfitting to single batch

- **`num_minibatches: 4`**: Moderate batch size
  - Allows multiple gradient updates per rollout
  - Balances variance (more batches) vs. stability (larger batches)

- **`gamma: 0.99`**: High discount factor
  - Long-term rewards matter (finding cave is delayed reward)
  - Encourages exploration for distant goals

- **`gae_lambda: 0.95`**: Standard GAE
  - Reduces variance in advantage estimates
  - Important for sparse reward environments

### Training Considerations

**Challenges:**
1. **Sparse Rewards**: BASALT environments give rewards only when task completed
2. **Long Episodes**: Finding caves can take many steps
3. **Large State Space**: High-dimensional image observations
4. **Hybrid Actions**: Must learn both discrete and continuous control

**Solutions:**
- GAE for better advantage estimation
- Entropy bonus for exploration
- CNN for efficient feature learning
- Hybrid architecture for action space

**Monitoring:**
- TensorBoard logs in `runs/` directory
- Key metrics: episodic return, policy loss, value loss, entropy
- Watch for: increasing returns (learning), stable losses (convergence)
- Videos saved to `videos/{run_name}/` directory (if `--capture-video` enabled)
- Action space diagnostics printed at startup

**Video Recording:**
- Enable with `--capture-video` flag
- Videos automatically saved during training
- Useful for monitoring agent behavior and progress
- Saved to `videos/` directory with run name

## Model Saving and Loading

### Saving Models During Training

Models are automatically saved during training:

1. **Best Model**: Saved whenever a new best episode return is achieved
   - File: `checkpoints/best_model_{run_name}.pth`
   - Contains: Model weights, optimizer state, training metadata

2. **Periodic Checkpoints**: Saved every N updates (default: 100)
   - File: `checkpoints/checkpoint_{update}_{run_name}.pth`
   - Useful for resuming training or comparing different training stages

3. **Final Model**: Saved at the end of training
   - File: `checkpoints/final_model_{run_name}.pth`
   - Final state after all training steps

**Checkpoint Contents:**
- `agent_state_dict`: Policy and value network weights
- `optimizer_state_dict`: Optimizer state (for resuming training)
- `global_step`: Number of training steps
- `episode_return`: Best reward achieved (for best model)
- `args`: Training hyperparameters
- `button_keys`, `num_button_actions`, `camera_dim`: Action space info

**Training Options:**
```bash
# Disable model saving
python train.py --save-model False

# Change save interval (save every 50 updates)
python train.py --save-interval 50

# Custom checkpoint directory
python train.py --checkpoint-dir my_checkpoints
```

### Loading Models for Inference

**Using the Inference Script:**

```bash
# Find your best model
ls checkpoints/best_model_*.pth

# Run inference with video recording
python inference.py --checkpoint checkpoints/best_model_MineRLBasaltFindCave-v0__train__1__1234567890.pth

# Run multiple episodes
python inference.py --checkpoint checkpoints/best_model_*.pth --num-episodes 5

# With rendering (see the agent play)
python inference.py --checkpoint checkpoints/best_model_*.pth --render

# Headless mode
xvfb-run -a python inference.py --checkpoint checkpoints/best_model_*.pth

# Custom output directory
python inference.py --checkpoint checkpoints/best_model_*.pth --output-dir my_results
```

**Programmatic Loading:**

```python
import torch
import gym
from train import Agent, extract_pov

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model_*.pth', map_location='cpu')

# Create environment and agent
env = gym.make('MineRLBasaltFindCave-v0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = Agent(env, device)
agent.load_state_dict(checkpoint['agent_state_dict'])
agent.eval()  # Set to evaluation mode

# Run inference
obs = env.reset()
with torch.no_grad():
    obs_tensor = torch.Tensor(extract_pov(obs)).to(device).unsqueeze(0)
    (action_discrete, action_camera), _, _, _ = agent.get_action_and_value(obs_tensor)
    action_dict = agent.action_to_dict(action_discrete, action_camera)
    obs, reward, done, info = env.step(action_dict)
```

## Next Steps

The PPO implementation provides a foundation for:
1. **Hyperparameter Tuning**: Adjust learning rate, entropy coefficient, etc.
2. **Architecture Improvements**: Deeper CNNs, attention mechanisms
3. **Advanced Techniques**: Curiosity-driven exploration, reward shaping
4. **Transfer Learning**: Pre-train on easier tasks, fine-tune on BASALT
5. **Evaluation**: Test trained models, compare with baselines

## Resources

- [MineRL Documentation](https://minerl.readthedocs.io/)
- [MineRL GitHub](https://github.com/minerllabs/minerl)
- [BASALT Challenge](https://www.aicrowd.com/challenges/neurips-2022-minerl-basalt-competition)
- [Gym Documentation](https://www.gymlibrary.dev/)

## License

This project is for educational and research purposes. MineRL and Minecraft have their own licenses.

## Notes

- First run will download Minecraft assets (may take time)
- Environment initialization takes 10-30 seconds
- Videos are saved in `./videos/` directory with timestamps
- Logs are saved in `./logs/` directory

