### Udacity Deep Reinforcement Learning Nanodegree:
# Project 1: Navigation
##Introduction
Let there be an agent sitting in the center of a large square world cluttered with yellow and blue bananas. The goal of the agent is to collect as many of the yellow bananas as possible while avoiding the blue ones. For this, the agent can move forward or backward and turn left or right. 

This repository is an implementation of a simplified version of the Banana Collector environment of the [Unity ML Agents Toolkit](https://github.com/Unity-Technologies/ml-agents)  with only one agent and no obstacles.
![Short sequency of the trained agent in action](trained_agent.gif)
##Project Details
####Project Environment
The project is implemented as a 4 layer neural network. The network is specified in the file `model.py`. The agent is implemented in the file `dqn_agent.py`, and the notebook `Navigation.ipynb` provides the interactive code to train an untrained and run a trained agent. 
####State Space
The agent's field of view consists of 7 horizontal rays around its forward direction. For each ray, the distance and category of the observed object is recorded. The category is one of the following:
- yellow banana
- blue banana
- wall
- other agent (not used in this simplified version)
For each ray, the velocity of the agent in 2D is also recorded.
The observation space, therefore, consists of $7 \cdot 5 + 2 = 37$ possible input values. 

####Action Space
The action space has 4 dimensions corresponding to the 4 discrete actions the agent can choose from:
- `0`: move forward
- `1`: move backward
- `2`: turn left
- `3`: turn right

####Rewards
A reward of $+1$ is provided for collecting a yellow banana, and a reward of $-1$ for a blue one.
####Goal
The task is episodic. The agent must get an average score of $+13$ over 100 consecutive episodes.

##Getting Started
The repository was developed and tested in a 64-bit Windows 10 virtual machine running Ubuntu 18.04 on an Intel Core i7-7700 CPU with dual NVIDIA GeForce GTX 1080.  

The following packages had to be installed:
- `curl`: 
    ```commandline
    sudo apt install curl
    ```
- `git`: 
    ```commandline
    sudo apt install git
    ```
- `conda`:
    ```commandline
    curl -O https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh
    bash Anaconda3-5.3.0-Linux-x86_64.sh
    ``` 

####Dependencies
This repository requires Python 3.6. A virtual environment `drlnd` was created like so:
```commandline
conda create -n drlnd python=3.6
```
Next, a minimal version of `openai gym` had to be installed:
```commandline
git clone https://github.com/openai/gym.git
cd gym
conda activate drlnd
pip install -e .
pip install -e '.[classic_control]'
pip install -e '.[box2d]'

```
To install the Udacity Deep Reinforcement Learning repository, the following command was used:
```commandline
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```
To add the `drlnd` environment to the `jupyter notebook` kernels, the following command was used:
```commandline
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

Here is the list of installed `python` packages in the `drlnd` environment:
```text
# packages in environment at /home/robond/anaconda3/envs/drlnd:
#
# Name                    Version                   Build  Channel
absl-py                   0.6.1                     <pip>
astor                     0.7.1                     <pip>
atomicwrites              1.2.1                     <pip>
attrs                     18.2.0                    <pip>
backcall                  0.1.0                     <pip>
bleach                    1.5.0                     <pip>
box2d-py                  2.3.5                     <pip>
ca-certificates           2018.03.07                    0    anaconda
certifi                   2018.10.15               py36_0    anaconda
chardet                   3.0.4                     <pip>
cycler                    0.10.0                    <pip>
decorator                 4.3.0                     <pip>
defusedxml                0.5.0                     <pip>
docopt                    0.6.2                     <pip>
entrypoints               0.2.3                     <pip>
future                    0.17.1                    <pip>
gast                      0.2.0                     <pip>
grpcio                    1.12.1           py36hdbcaa40_0    anaconda
grpcio                    1.16.0                    <pip>
grpcio                    1.11.0                    <pip>
html5lib                  0.9999999                 <pip>
idna                      2.7                       <pip>
ipykernel                 5.1.0                     <pip>
ipython                   7.1.1                     <pip>
ipython-genutils          0.2.0                     <pip>
ipywidgets                7.4.2                     <pip>
jedi                      0.13.1                    <pip>
Jinja2                    2.10                      <pip>
jsonschema                2.6.0                     <pip>
jupyter                   1.0.0                     <pip>
jupyter-client            5.2.3                     <pip>
jupyter-console           6.0.0                     <pip>
jupyter-core              4.4.0                     <pip>
kiwisolver                1.0.1                     <pip>
libedit                   3.1.20170329         h6b74fdf_2  
libffi                    3.2.1                hd88cf55_4  
libgcc-ng                 8.2.0                hdf63c60_1  
libstdcxx-ng              8.2.0                hdf63c60_1  
Markdown                  3.0.1                     <pip>
MarkupSafe                1.0                       <pip>
matplotlib                3.0.1                     <pip>
mistune                   0.8.4                     <pip>
more-itertools            4.3.0                     <pip>
nbconvert                 5.4.0                     <pip>
nbformat                  4.4.0                     <pip>
ncurses                   6.1                  hf484d3e_0  
notebook                  5.7.0                     <pip>
numpy                     1.15.3                    <pip>
openssl                   1.1.1                h7b6447c_0    anaconda
pandas                    0.23.4                    <pip>
pandocfilters             1.4.2                     <pip>
parso                     0.3.1                     <pip>
pexpect                   4.6.0                     <pip>
pickleshare               0.7.5                     <pip>
Pillow                    5.3.0                     <pip>
pip                       18.1                     py36_0  
pip                       18.1                      <pip>
pluggy                    0.8.0                     <pip>
prometheus-client         0.4.2                     <pip>
prompt-toolkit            2.0.7                     <pip>
protobuf                  3.5.2                     <pip>
ptyprocess                0.6.0                     <pip>
py                        1.7.0                     <pip>
pyglet                    1.3.2                     <pip>
Pygments                  2.2.0                     <pip>
PyOpenGL                  3.1.0                     <pip>
pyparsing                 2.3.0                     <pip>
pytest                    3.9.3                     <pip>
python                    3.6.7                h0371630_0  
python-dateutil           2.7.5                     <pip>
pytz                      2018.7                    <pip>
PyYAML                    3.13                      <pip>
pyzmq                     17.1.2                    <pip>
qtconsole                 4.4.2                     <pip>
readline                  7.0                  h7b6447c_5  
requests                  2.20.0                    <pip>
scipy                     1.1.0                     <pip>
Send2Trash                1.5.0                     <pip>
setuptools                40.5.0                   py36_0  
six                       1.11.0                    <pip>
six                       1.11.0                   py36_1    anaconda
sqlite                    3.25.2               h7b6447c_0  
tensorboard               1.7.0                     <pip>
tensorflow                1.7.1                     <pip>
termcolor                 1.1.0                     <pip>
terminado                 0.8.1                     <pip>
testpath                  0.4.2                     <pip>
tk                        8.6.8                hbc83047_0  
torch                     0.4.0                     <pip>
tornado                   5.1.1                     <pip>
traitlets                 4.3.2                     <pip>
unityagents               0.4.0                     <pip>
urllib3                   1.24                      <pip>
wcwidth                   0.1.7                     <pip>
Werkzeug                  0.14.1                    <pip>
wheel                     0.32.2                   py36_0  
widgetsnbextension        3.4.2                     <pip>
xz                        5.2.4                h14c3975_4  
zlib                      1.2.11               ha838bed_2  
```
###Udacity Project Repository
Download the [project's repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) from Udacity's GitHub page if you like to re-implement the project yourself.
The environment can be downloaded [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip). The project's GitHub page contains links to download it for operating systems other than Linux.
##Instructions
Make sure the `Banana.x86_64` and the folder `Banana_Data` from your environment are in your project directory, together with the `model.py` and the `dqn_agent.py` files and the `Navigation.ipynb` notebook:
```text
Banana_Data
Banana.x86_64
checkpoint.pth
dqn_agent.py
model.py
Navigation.ipynb
__pycache__
readme.md
unity-environment.log
```
The files `checkpoint.pth` and `unity-environment.log` are (re-)created when running the notebook and don't exist initially.

####Running the Code
To start the notebook, open a terminal and navigate to your project directory or a parent thereof, then enter
````commandline
jupyter notebook
````
The notebook is opened in your standard browser. You might have to navigate to the project directory, then start `Navigation.ipynb`.
Run the first three cells by clicking `SHIFT ENTER`
####Training the Agent
Define the average score to be reached. The project required $13$ or more, $15$ was selected. Then run the relevant cells. Training progress is printed every $100$ episodes, moving average score has reached the predefined threshold, training is complete. The weights are then written to the file `checkpoint.pth`. and the average score over the last $100$ timesteps is plotted.
####Running the Agent
To run the trained agent, load the weights from `checkpoint.pth`, reset the environment with `train_mode=False` and the score reset to $0$, then run until done.
