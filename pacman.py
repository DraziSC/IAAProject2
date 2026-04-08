import game_engine
import agents
import numpy as np

if __name__ == "__main__":
    #ghost order: 'Blinky', 'Pinky', 'Inky', 'Clyde'

    #Test
    #pacman_policy = agents.keyboard_controller #use the arrow keys to control pacman
    #ghost_policies = [agents.random_walk for _ in range(4)] 
    #frightened_ghost_policies = [agents.random_walk for _ in range(4)]
    #game_engine.main(pacman_policy, ghost_policies, frightened_ghost_policies, map_file='maps/originalClassic-single-ghost.txt')

    #---TP2---
    map_file='maps/originalClassic-single-ghost.txt'
    ghost_true_prob = 0.2 # probability of the ghost sensor yielding an accurate reading
    n_experiments = 10
    
    score = np.zeros(n_experiments)
    for i in range(n_experiments):
        pacman_policy = agents.pacmanHMM
        ghost_policies = [agents.inky_agent, agents.blinky_agent, agents.pinky_agent, agents.inky_agent, agents.clyde_agent]
        frightened_ghost_policies = [agents.random_walk for _ in range(4)]
        score[i] = game_engine.main(pacman_policy, ghost_policies, frightened_ghost_policies, map_file=map_file, ghost_true_prob=ghost_true_prob)
    print(f"Average score of {n_experiments} experiments: ", np.mean(score))
