# DRL_Navigation
This project is a part of:  
 [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
 )

The project uses DDPG algorithm to solve 'Reacher Arm' environment.  

Goal is to move 20 double-jointed arms to be in target position as long as possible. A reward of +0.1 is provided for each step that the agent's hand is in the goal location.

![reacher app](./data/reacher.gif)

# Environment details
* There were 20 arms. 
* Each arm contain 33 state observations
* Action space containd 4 contunuous values in range from -1 to 1.
* Each frame with arm in good location yields 0.1 reward
* Each episode has 1000 frames
* Overall goal is to train agent to keep all arms in target location for 100 epochs with average total score of 30 

# Requirements
Below you can find a list of requirements required to run train.py & play.py scripts
## Resources
- python 3.6
- Reacher app (this is delivered by Udacity Team) - I was using [headless linux client](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to be able to run it seamlesly on any environment (even without display)

## Python packages
- torch 
- numpy 
- tqdm
- matplotlib
- unityagents

# Usage
## Training
```bash
python3 train.py
```

 # Details
Implementation approach & details and metrics can be found in [report](./Report.md) file.

Here is quick look for the results:  
![data](./data/results_DDPG%20128+128_20200904_224050.png)  
Link to [full report](./Report.md) 