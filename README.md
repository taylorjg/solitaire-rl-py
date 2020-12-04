# Description

Solve [Peg solitaire](https://en.wikipedia.org/wiki/Peg_solitaire)
using reinforcement learning. The code is written in Python and uses:

* [PyTorch](https://pytorch.org/)
* [OpenAI Gym](https://gym.openai.com/)
* A custom OpenAI Gym environment: [gym-solitaire](https://github.com/taylorjg/gym-solitaire)

# How it works  

These are just some rough notes for now.

## Game details

* Number of board positions: 33
* Number of all possible actions: 76
  * But, only a small subset of actions are valid in each state
  * e.g. the initial state has 4 valid actions

## Environment details

* State/observations:
  * Are numpy arrays of 33 `0`s/`1`s
    * `0` for an empty board position
    * `1` for an occupied board position
* Rewards:
  * 0 for all moves except the final move of each episode
  * Final move of each episode:
    * +100 if the puzzle is solved correctly i.e. there is a single remaining piece located at the centre of the board
    * Otherwise, the negative sum of the Manhattan distances (from the centre of the board) of the remaining pieces  

## Agent details

* The neural network approximates the afterstate value function
  * It has 33 inputs and 1 output
  
## Training loop details  

* episodes loop:
  * reset env
  * env steps loop:
    * evaluate valid actions for the current state
      * get the valid actions from the env (not standard Gym behaviour)
      * figure out the next board state for each valid action
      * push all the next board states through the neural network to predict the afterstate values
    * use the policy to choose an action
      * with epsilon prob: random choice
      * with (1 - epsilon) prob: greedy choice i.e. choose the action with the highest afterstate value estimate
    * step the env with the chosen action
    * calculate the target afterstate value for the current state based on the reward from the env
    added to the discounted estimate of the value for the afterstate resulting from taking the action
    * calculate the loss i.e. the difference between the current estimate of the afterstate
    value of the current state and the calculated target value
    * back propagate the loss  

# Training plots

![Training plots](training_plots.png)

# Setting up

I'm pretty new to Python and Conda etc. but I think the following should do it:

```
conda env create -f environment.yml
conda develop ../gym-solitaire
```

This assumes that you have cloned [gym-solitaire](https://github.com/taylorjg/gym-solitaire) into `../gym-solitaire`. 

# Play

The following command will load a previously trained model and play a single episode of Solitaire:

```
python td_solitaire.py --play
```

# Train

The following command will train a model and, if successful, save the trained model to `td_solitaire.pt`:

```
python td_solitaire.py
```

# Links

* [Peg solitaire](https://en.wikipedia.org/wiki/Peg_solitaire)
* I created a custom [OpenAI Gym](https://gym.openai.com/) environment: 
  * [gym-solitaire](https://github.com/taylorjg/gym-solitaire)
  * [How to create new environments for Gym](https://github.com/openai/gym/blob/master/docs/creating-environments.md)  
* To solve Solitaire using reinforcement learning, I used ideas from:
  * [Reinforcement Learning in the Game of Othello:
Learning Against a Fixed Opponent
and Learning from Self-Play](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/paper-othello.pdf)
  * _TODO_: list other sources...
