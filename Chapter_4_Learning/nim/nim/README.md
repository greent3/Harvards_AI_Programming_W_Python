# OBJECTIVE
In the game Nim, we begin with some number of piles, each with some number of objects. 
Players take turns: on a player’s turn, the player removes any non-negative number of objects from any one non-empty pile. Whoever removes the last object loses.
In this problem, we’ll build an AI to learn the strategy for this game through reinforcement learning. 
By playing against itself repeatedly and learning from experience, eventually our AI will learn which actions to take and which actions to avoid.
We do this through Q-learning, or assigning a reward value for every (state, action) pair.
We’ll represent the game as an array (game board) of numbers (# of pieces in each row). 

# MY RESPONSIBILITIES
I completed the logic and implementation of the following functions in nim.py:
- get_q_value() 
- update_q_value() 
- best_future_reward() 
- choose_action()

All other functions and play.py were provided by Brian Yu over at Harvard. 
