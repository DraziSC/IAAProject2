import game_engine
import pygame
import pacman_perceptions
import random
import numpy as np
import matplotlib.pyplot as plt




def keyboard_controller(game_state):
    direction = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_state['running'] = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                direction = 'up'
            elif event.key == pygame.K_DOWN:
                direction = 'down'
            elif event.key == pygame.K_LEFT:
                direction = 'left'
            elif event.key == pygame.K_RIGHT:
                direction = 'right'
                
    pacman = game_state['pacman']
    grid = game_state['grid']
    grid_size = game_state['grid_size']
    
    if (direction is None and not game_engine.PACMAN_CONTINUOUS_MOTION) or direction in game_engine.get_valid_directions((pacman['x'],pacman['y']), grid, grid_size):
        pacman['previous_direction'] = pacman['direction']
        pacman['direction'] = direction
            
    elif direction is not None and direction not in game_engine.get_valid_directions((pacman['x'],pacman['y']), grid, grid_size):
        pacman['next_direction'] = direction
        
    elif direction is None and game_engine.PACMAN_CONTINUOUS_MOTION and pacman['next_direction'] in game_engine.get_valid_directions((pacman['x'],pacman['y']), grid, grid_size):
        pacman['previous_direction'] = pacman['direction']
        pacman['direction'] = pacman['next_direction']
        pacman['next_direction'] = None
        
def random_walk(agent, game_state):
    grid = game_state['grid']
    grid_size = game_state['grid_size']
    directions = game_engine.get_valid_directions((agent['x'],agent['y']), grid, grid_size)
    if len(directions)>1 and agent['direction'] is not None and game_engine.opposite_direction(agent['direction']) in directions:
        directions.remove(game_engine.opposite_direction(agent['direction']))

    agent['direction'] = random.choice(directions)
    

def stationary_agent(ghost, game_state):
    pass
    
def blinky_agent(ghost, game_state):
    #moves towards pacman
    pacman = game_state['pacman']
    grid = game_state['grid']
    grid_size = game_state['grid_size']

    directions = game_engine.get_valid_directions((ghost['x'],ghost['y']), grid, grid_size)    
    op = game_engine.opposite_direction(ghost['direction'])
    if len(directions)>1 and op in directions:    
        directions.remove(op)
    
    curr_pos = (ghost['x'],ghost['y'])
    best_pos = game_engine.compute_new_pos(curr_pos, directions[0])
    best_dist = game_engine.manhattan_distance(best_pos, (pacman['x'],pacman['y']))
    best_dir = directions[0]
    
    for dir in directions:
        cand_pos = game_engine.compute_new_pos(curr_pos, dir)
        cand_dist = game_engine.manhattan_distance(cand_pos, (pacman['x'],pacman['y']))
        
        if cand_dist < best_dist:
            best_dist = cand_dist
            best_pos = cand_pos
            best_dir = dir
    ghost['direction'] = best_dir
    
    
def pinky_agent(ghost, game_state):
    #moves to 4 cells in front of pacman
    pacman = game_state['pacman']
    grid = game_state['grid']
    grid_size = game_state['grid_size']

    target_pos = game_engine.compute_new_pos((pacman['x'],pacman['y']), pacman['direction'], 4)

    directions = game_engine.get_valid_directions((ghost['x'],ghost['y']), grid, grid_size)    
    op = game_engine.opposite_direction(ghost['direction'])
    if len(directions)>1 and op in directions:    
        directions.remove(op)
    
    curr_pos = (ghost['x'],ghost['y'])
    best_pos = game_engine.compute_new_pos(curr_pos, directions[0])
    best_dist = game_engine.manhattan_distance(best_pos, target_pos)
    best_dir = directions[0]
    
    for dir in directions:
        cand_pos = game_engine.compute_new_pos(curr_pos, dir)
        cand_dist = game_engine.manhattan_distance(cand_pos, target_pos)
        
        if cand_dist < best_dist:
            best_dist = cand_dist
            best_pos = cand_pos
            best_dir = dir
    ghost['direction'] = best_dir

    
def inky_agent(ghost, game_state):
    random_walk(ghost, game_state)
    
def clyde_agent(ghost, game_state):
    #moves towards pacman if far, otherwise moves randomly
    pacman = game_state['pacman']
    dist = game_engine.manhattan_distance((ghost['x'],ghost['y']), (pacman['x'],pacman['y']))
    if dist > 5:
        blinky_agent(ghost, game_state)
    else:
        random_walk(ghost, game_state)

def run_away_from_pacman(ghost, game_state):
    pacman = game_state['pacman']
    grid = game_state['grid']
    grid_size = game_state['grid_size']

    directions = game_engine.get_valid_directions((ghost['x'],ghost['y']), grid, grid_size)    
    op = game_engine.opposite_direction(ghost['direction'])
    if len(directions)>1 and op in directions:    
        directions.remove(op)
        
    curr_pos = (ghost['x'],ghost['y'])
    best_pos = game_engine.compute_new_pos(curr_pos, directions[0])
    best_dist = game_engine.manhattan_distance(best_pos, (pacman['x'],pacman['y']))
    best_dir = directions[0]
    
    for dir in directions:
        cand_pos = game_engine.compute_new_pos(curr_pos, dir)
        cand_dist = game_engine.manhattan_distance(cand_pos, (pacman['x'],pacman['y']))
        
        if cand_dist > best_dist:
            best_dist = cand_dist
            best_pos = cand_pos
            best_dir = dir
    ghost['direction'] = best_dir
    
    

#---TP2---

def get_neighbours(s, valid_positions):
    #returns the valid cells (without obstacles) adjacent to the cell s
    valid_neighbours = []
    xs, ys = valid_positions[s]
    for x in (xs-1, xs, xs+1):
        for y in (ys-1, ys, ys+1):
            if (x,y) != (xs,ys) and (x, y) in valid_positions:
                valid_neighbours.append(valid_positions.index((x,y)))
    return valid_neighbours

def visualise_belief(belief, cax):
    #Display a single, dynamically updating heatmap of a belief distribution.
    cax.set_data(belief)
    cax.figure.canvas.draw()
    cax.figure.canvas.flush_events()
        
def vector_to_matrix(belief, grid_size, valid_positions):
    #Convert a 1D belief vector into a 2D matrix for visualization.
    belief_matrix = np.zeros(grid_size)
    for i in range(len(belief)):
        x, y = valid_positions[i]
        belief_matrix[y][x] = belief[i]
    return belief_matrix


def calculate_observation_probability(valid_positions,s, o, true_prob):
    #TODO: Implement the sensor model P(O_t | X_t) for the ghost
    #s = ghost state, o = observation
    #true_prob is the probability of the sensor yielding an accurate reading
    #should return a single probability value of the sensor reading given the ghost's position    
    pass

def bayesian_filter(observation, model):
    #TODO: Implement the Bayesian filter to update the belief vector.
    
    #1. PREDICTION: Compute the belief using transition model

    #2. UPDATE: Incorporate evidence using observation model
    
    #3. NORMALIZATION: Ensure the belief map is a probability distribution
    pass

def pacmanHMM(game_state):
    if game_state['pacman']['model'] is None:
        model = {}
        model['num_observations'] = model['num_states'] = len(game_state['valid_positions'])

        #TODO: complete the code below to initialise the model. The belief vector is already provided with a uniform prior.
        
        #define the belief vector. To simplify, we will use a 1D vector of size num_states 
        #(i.e., the flattened grid of cells with no obstacles)
        model['belief'] = np.full(model['num_states'], 1.0/model['num_states']) #1D initial state distribution.

        #1.Define the Transition Matrix (T) - models the probability of moving from one state to any other state
        
        #2.Define the observation Matrix (E) - models the likelihood of sensor readings given the ghost's position
        
        ##visualisation
        if game_engine.VISUALISE:
            plt.close('all')
            plt.ion()
            fig, ax = plt.subplots()
            cax = ax.imshow(vector_to_matrix(model['belief'], game_state['grid_size'], game_state['valid_positions']), cmap='hot', interpolation='nearest')
            plt.colorbar(cax, label='Probability')
            ax.set_title('Ghost Belief')
            plt.pause(0.001)
            model['fig'] = fig
            model['ax'] = ax
            model['cax'] = cax
            
        game_state['pacman']['model'] = model    

    #get the noisy ghost observation
    observation = game_state['valid_positions'].index(pacman_perceptions.noisy_ghost_position_sensor(game_state, 0, game_state['pacman']['ghost_true_prob']))

    #apply the bayesian filter to update the belief
    bayesian_filter(observation, game_state['pacman']['model'])

    if game_engine.VISUALISE:
        visualise_belief(vector_to_matrix(game_state['pacman']['model']['belief'], game_state['grid_size'], game_state['valid_positions']), game_state['pacman']['model']['cax'])

    #TODO: Use the updated belief in a reactive agent to move Pac-Man
    #replace this line by your own implementation    
    random_walk(game_state['pacman'], game_state)
            
            

    
    
