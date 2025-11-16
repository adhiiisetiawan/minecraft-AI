# Minecraft AI

A project for experimenting with reinforcement learning in Minecraft using the MineRL environment. This project includes scripts for running random actions and executing predefined action sequences with video recording capabilities.

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

### Step 1: Create Conda Environment

```bash
# Create a new conda environment with Python 3.11
conda create -n minecraft-ai python=3.11 -y

# Activate the environment
conda activate minecraft-ai
```

### Step 2: Install Dependencies

```bash
# Install pip dependencies
pip install gym
pip install minerl
pip install numpy==1.23.5  # Important: numpy 1.23.5 for Python 3.11 compatibility
pip install opencv-python
pip install tqdm
```

**Note**: If you encounter issues with numpy installation, you may need to install it with:
```bash
pip install numpy==1.23.5 --no-build-isolation
```

### Step 3: Verify Installation

```bash
python -c "import gym; import minerl; print('Installation successful!')"
```

## Project Structure

```
minecraft-AI/
├── README.md              # This file
├── getting_started.py     # Random action execution script
├── sequence_action.py     # Scripted action sequence with video recording
├── videos/                # Output directory for recorded videos
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

## Next Steps

To implement reinforcement learning:

1. **Choose an RL algorithm**: PPO, DQN, A3C, etc.
2. **Design observation processing**: Process the `pov` images (CNN encoder)
3. **Handle hybrid action space**: Use policy networks that output both discrete and continuous actions
4. **Training loop**: Implement experience collection and policy updates
5. **Evaluation**: Test trained models on the environment

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

