# Restricted-MaxEnt-Meta-RL
## Instructions 


To install locally, you will need to first install MuJoCo. For the task distributions in which the reward function varies (Cheetah, Ant, Humanoid), install MuJoCo200. Set LD_LIBRARY_PATH to point to both the MuJoCo binaries (/$HOME/.mujoco/mujoco200/bin) as well as the gpu drivers (something like /usr/lib/nvidia-390, you can find your version by running nvidia-smi). For the remaining dependencies, we recommend using miniconda - create our environment with conda env create -f docker/environment.yml This installation has been tested only on 64-bit Ubuntu 16.04.

For the task distributions where different tasks correspond to different model parameters (Walker and Hopper), MuJoCo131 is required. Simply install it the same way as MuJoCo200. These environments make use of the module rand_param_envs which is submoduled in this repository. Add the module to your python path, export PYTHONPATH=./rand_param_envs:$PYTHONPATH (Check out direnv for handy directory-dependent path managenement.)

Experiments are configured via json configuration files located in ./configs. To reproduce an experiment, run: python launch_experiment.py ./configs/[EXP].json

By default the code will use the GPU - to use CPU instead, set use_gpu=False in the appropriate config file.

Output files will be written to ./output/[ENV]/[EXP NAME] where the experiment name is uniquely generated based on the date. The file progress.csv contains statistics logged over the course of training. We recommend viskit for visualizing learning curves: https://github.com/vitchyr/viskit

Network weights are also snapshotted during training. To evaluate a learned policy after training has concluded, run sim_policy.py. This script will run a given policy across a set of evaluation tasks and optionally generate a video of these trajectories. Rendering is offline and the video is saved to the experiment folder.
