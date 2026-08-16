import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

# Import stats class for report.
from stats import Stats

# Given a state representation of 1(xy)(xy)
# one splits into two (xy), bot and cat
# two splits into four (x,y), their coordinates
# return both row and col of bot and cat
def state_split(s: int):
    bot_pos, cat_pos = divmod(s, 100)
    return bot_pos, cat_pos

#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    ### HYPERPARAMS SECTION
    
    # Alpha and Gamma
    learn_rate = 0.18
    disc_factor = 0.95

    # Epsilon, it's minumum and decay
    exp_rate = 1.0
    min_exp_rate = 0.01
    
    # Decay around the 3000th mark (or 60%), then optimize/exploit for all learned states instead.
    decay_type = "lin"
    exp_rate_decay = 0.9985
    lin_rate_decay = 0.0003

    # Allot bajillion moves
    MAX_MOVES = 500
    
    # Stats counters
    stats = Stats(window=500, decay_type=decay_type)
    success_count = 0
    episode_steps_list = []
    episode_rewards_list = []







    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
        
        state, _ = env.reset()
        
        # Initialize episode params
        steps = 0     
        episode_reward = 0.0
        
        done = False
        truncated = False
        
        # Bot pos for checking wallbangs or circling around that result to fruitlessness 
        prev_bot_pos = None
        prev_prev_bot_pos = None

        while not done and not truncated:
                
            # Action Step
            if random.random() < exp_rate:
                action = env.action_space.sample()
            else:
                q_values = q_table[state]
                best_actions = np.flatnonzero(q_values == np.max(q_values))
                action = int(random.choice(best_actions))
                
            new_state, _, terminated, truncated, _ = env.step(action)
            done = terminated
            steps += 1

            if steps >= MAX_MOVES and not done:
                truncated = True
                
            bot_pos, cat_pos = state_split(state)
            new_bot_pos, new_cat_pos = state_split(new_state)
            player_moved = bot_pos != new_bot_pos
            cat_moved = cat_pos != new_cat_pos
            
            # Reward Section
            # counter to reduce time (i.e. a sol)
            reward = -2.0 
            
            if done:
                reward += 100.0
            elif truncated:
                reward += -300.0
            else:
                if not cat_moved:
                    if not player_moved:
                        reward -= 10.0
                    elif prev_prev_bot_pos == new_bot_pos:
                        reward -= 8.0
                        
            prev_prev_bot_pos = prev_bot_pos
            prev_bot_pos = bot_pos
            episode_reward += reward

            q_value = q_table[state][action]

            if done or truncated:
                target = reward
            else:
                target = (reward + disc_factor * np.max(q_table[new_state]))

            td_error = target - q_value
            q_table[state][action] += learn_rate* td_error
            
            state = new_state
            # END OF MOVE LOOP

        # DECAY EPSILON AFTER NEW EPISODE
        if decay_type == "lin":
            exp_rate = max(min_exp_rate, exp_rate - lin_rate_decay)

        elif decay_type == "exp":
            exp_rate = max(min_exp_rate, exp_rate * exp_rate_decay)

        # STATS SECTION
        if done:
            success_count += 1
        episode_steps_list.append(steps)
        episode_rewards_list.append(episode_reward)
        stats.add_episode(episode_reward, steps)
        stats.add_result(captured=done,truncated=truncated)

        if ep == 1 or ep % 500 == 0 or ep == episodes:
            ave_steps = np.mean(episode_steps_list[-100:])
            ave_reward = np.mean(episode_rewards_list[-100:])
            success_rate = success_count / ep

            stats.add_summary(
                episode=ep,
                ave_steps=ave_steps,
                ave_reward=ave_reward,
                success_rate=success_rate,
                epsilon=exp_rate,
            )

        # RUN AFTER ALL EPISODE FINISHED 
        if ep == episodes:
            stats.plot(cat_name)
                


















        
        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)
    
    return q_table