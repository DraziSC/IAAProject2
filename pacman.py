import random
import game_engine
import agents
import numpy as np
import random

if __name__ == "__main__":
    #ghost order: 'Blinky', 'Pinky', 'Inky', 'Clyde'

    #Test
    #pacman_policy = agents.keyboard_controller #use the arrow keys to control pacman
    #ghost_policies = [agents.random_walk for _ in range(4)] 
    #frightened_ghost_policies = [agents.random_walk for _ in range(4)]
    #game_engine.main(pacman_policy, ghost_policies, frightened_ghost_policies, map_file='maps/originalClassic-single-ghost.txt')

    #---TP2---

    map_file='maps/originalClassic-single-ghost.txt'
    ghost_true_prob = 0.8 # probability of the ghost sensor yielding an accurate reading
    ghost_true_prob_array = [0.1, 0.25, 0.5, 0.75, 0.9]
    #ghost_true_prob_array = [0.9] 

    n_experiments = 10
    seed = 42
               
    score = np.zeros(n_experiments)
    for j in range(len(ghost_true_prob_array)):
        ghost_true_prob = ghost_true_prob_array[j]  
        wins = 0
        for i in range(n_experiments):
            random.seed(seed + i)
            print(f"Experiment {i+1}/{n_experiments} with ghost_true_prob: {ghost_true_prob} seed: {seed + i}")
            pacman_policy = agents.pacmanHMM
            #ghost_policies = [agents.inky_agent, agents.blinky_agent, agents.pinky_agent, agents.clyde_agent]
            ghost_policies = [agents.blinky_agent]
            #ghost_policies = [agents.pinky_agent]
            #ghost_policies = [agents.inky_agent]
            #ghost_policies = [agents.clyde_agent]
            frightened_ghost_policies = [agents.random_walk for _ in range(4)]
            
            score[i], won = game_engine.main(pacman_policy, ghost_policies, frightened_ghost_policies, map_file=map_file, ghost_true_prob=ghost_true_prob)
            wins+=won

        print(f"Average score of {n_experiments} experiments: ", np.mean(score), "with ghost_true_prob: ", ghost_true_prob,"Wins:", wins)
        
            
