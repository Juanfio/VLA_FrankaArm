# VLA-FrankaArm: Vision-Language-Action Manipulation with a Franka Arm

Vision-Language-Action Model (VLA) trained on Open X Embodiment (OxE) data for manipulation tasks with Franka Arm. Policy rollout on simulated environment.

> 🚧 Early-stage research project. Installation and usage instructions coming soon.

<!-- # Description

* Trained VLA model using OxE datasets (Freiburg Franka Play & Stanford Hydra) of Franka Arm. The trained model is fine-tuned to solve manipulation task with a Franka Arm in a simulated environment. -->

<!-- # Demo -->
<!-- Add some screenshots of simulated and OxE data -->


<!-- # Installation -->

# Usage

`Select datasets via cfg.json`

```
conda activate oxe_download
python -m preprocessing.download_oxe_data --settings cfg_download_oxe_data.json
```

```
conda activate robotic
python -m model.train --settings cfg.json --n_epochs 5
```

<!-- SOMETHING ABOUT PRINTING PLOTS -->

<!-- SOMETHING ABOUT RUNNING FINE-TUNE -->

<!-- SOMETHING ABOUT ROLLOUT -->

# Project Structure

- `/preprocessing/download_oxe_data.py`: `python download_oxe_data.py --settings cfg.json` for downloading OxE dataset.
- `/preprocessing/preprocessing.py`: Run the preprocessing pipeline: 
    1. Discretize action space
    2. Resize images
    3. Embedding language instruction
    4. Create dataloaders.
- `/model/models.py`: Vision Transformer & Action Prediction models
- `/model/train.py`: Training loop. `python -m model.train --settings cfg.json --n_epochs 5`
- `/simulation/miniarm_env.py`: PyBullet simulation of the Franka Arm; goal is to reach a target sphere
- `/simulation/oracle.py`: generates rollout episodes and saves data to `/data`
- `/data`: generated episode data (OxE and simulated data)
- `/environments/oxe_download.yml`: For downloading the OxE data.
- `/environments/env_robotics.yml`: For model training.


<!-- # Simulation projects

* Vision Robotic's projects with simulated data (PyBullet) over multiple different tasks (e.g., project 1 is a reach task.)
* Projects aim to address the sim-to-real problem, i.e., policies learned in simulated environments often failed when deployed on real-world physical robots.

# Project structure

#### Project 1 - Reach Task

* /simulation:
    * miniarm_env.py: Environment with Franka Arm and sphere.
    * oracle.py: Simulate Frank Arm movement towards the sphere with privileged information; i.e., actions of the arm are taken in the direction of the sphere.
* /notebooks:
    * track_A_pybullet_fundamentals.ipynb.
        - examples showing how to build the simulated environment (with a Franka Arm).
    * track_B_pytorch_fundamentals.ipynb.
        - examples showing how to build the NN controller (with a Franka Arm).
    * track_C_pybullet_pytorch_integration.ipynb.
        - Simulate Franka Arm reaching a sphere.
        - Collect demonstrations.
        - Model predicting arm movement from visual (rgb) and proprioceptive inputs. Model description:
            * ImageEncoder(): Conv2d is used to encode the rgb data generated using p.getCameraImage() from PyBullet.
            * ProprioEncoder(): Linear layers are used to encode the proprioceptive data containing the angles and velocities of the Franka Arm joints (MiniArmEnv()_get_proprio())
            * ActionPrediction(): Encodes the rgb and proprioceptive inputs and outputs an action (delta from current end-effector position).
            * The model reaches the sphere position by generating small increments (deltas) from current end-effector position towards the sphere. It uses the image (rgb) and the proprio data to guide its actions.
* /data:
    - Simulated franka arm reaching a sphere in .hdf5 format.
* environment:
    env_robotics.yml

#### Project 2

_In progress_

#### Project 3

_In progress_ -->


# References

1. Billard, A. et al. A roadmap for AI in robotics. Nature Machine Intelligence 7, 818–824 (2025).
2. Liu, Y. et al. Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI. arXiv:2407.06886 (2024).
3. Roy, N. et al. From Machine Learning to Robotics: Challenges and Opportunities for Embodied Intelligence. arXiv:2110.15245 (2021).
4. Open X-Embodiment Collaboration et al. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. Proc. IEEE International Conference on Robotics and Automation (ICRA), 6892–6903 (2024).
5. Kim, M. J. et al. OpenVLA: An Open-Source Vision-Language-Action Model. Proc. 8th Conference on Robot Learning 270, 2679–2713 (2025).
6. Ha, D. & Schmidhuber, J. Recurrent World Models Facilitate Policy Evolution. Advances in Neural Information Processing Systems 31, 2451–2463 (2018).