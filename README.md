# RL-indoor-mobility-demo
Virtual indoor environment for evaluating visually-guided mobility using reinforcement learning

## About the project
This module was developed to evaluate and optimize image processing for prosthetic vision using reinforcement learning. For more background on automated optimization of image processing for prosthetic vision please refer to https://www.biorxiv.org/content/10.1101/2020.12.19.423601v1

Reference to the current work will be updated here shortly. 

> Jaap de Ruyter van Steveninck, Sam Danen, Burcu Küçükoğlu, Umut Güçlü, Richard van Wezel, Marcel van Gerven (2021). Reinforcement-learning based optimization of prosthetic vision for indoor mobility [Manuscript in preparation]

Credits for the code go to Sam Danen.


## Getting Started

### Prerequisites

- cv2
- socket

### Usage

Step 1: 
Start the environment server by running the Unity server application which can be found in the Build folder. In the GUI, press 'start' to accept the default environment parameters.

Step 2: 
For a demonstration of navigation in the environment run:

  ```sh
  python PythonScrips/demoNavigation.py
  ```
Controls:
- 1: normal vision
- 2: low resolution prosthethic vision
- 3: high resolution prosthetic vision
- w: forward
- a: left
- d: right
- r: reset
- q: quit

Step 3: 
For the usage and implementation, refer to PythonScrips/demoUsage.py
