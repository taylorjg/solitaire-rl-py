# Description

Solve [Peg solitaire](https://en.wikipedia.org/wiki/Peg_solitaire)
using reinforcement learning. The code is written in Python and uses:

* [PyTorch](https://pytorch.org/)
* [OpenAI Gym](https://gym.openai.com/)
* A custom OpenAI Gym environment: [gym-solitaire](https://github.com/taylorjg/gym-solitaire)

> _TODO: add a lot more description here..._

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
python td-solitaire.py --play
```

# Train

The following command will train a model and, if successful, save the trained model to `td-solitaire.pt`:

```
python td-solitaire.py
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
