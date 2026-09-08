# CSINTSY - MCO3
## Catbot

Given an 8x8 open field, you (as the player) is tasked to catch the cat from the opposing corner and... </br>
Wait no, it's Catbot, not Catperson, you (as the coder) is tasked to program a bot that can catch the cat. </br>
Each cat has its own set of behaviours and your goal is to be able to have the AI learn a policy for the cat's behavior. </br> 
The bot is tested for one final match against the cat's AI based from the knowledge built from 5000 episodes. </br></br>

Method of reinforcement learning used is Tabular Q-Learning.</br>

### Commandzzz:

To play the game yourself you can do:
```
python play.py --cat <catname>
```
</br>

To play the game with the bot:
```
python bot.py --cat <catname> --render n
```

</br>

where: </br>
catname: refers to cat's name found under the bot.py/play.py args. </br>
n: render an episode in pygame for each n (maximum)</br>

training.py contains the training module used in tabular q-learning. </br>
stats.py contains the statistics module used in displaying the records of episodes and its data. </br>

### Requirements
- Python (duh)
    - Python Packages (numpy, gymnasium, matplotlib, pygame).