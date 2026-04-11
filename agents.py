from pyexpat import model
from turtle import left, right

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
    
def visualise_belief_number(belief_n,cax_n):
    cax_n.clear() #clear the previous text annotations and prevents old frames from stacking on top of new ones
    cax_n.imshow(belief_n, cmap='hot', interpolation='bicubic', vmin=0, vmax=1)
    rows, cols = belief_n.shape
    for y in range(rows):
        for x in range(cols):
            if belief_n[y][x] > 0:
                cax_n.text(x, y, f"{belief_n[y][x]:.2f}", ha='center', va='center', color='white', fontsize=7)

    cax_n.set_title('Ghost Belief in Numbers')
    cax_n.figure.canvas.draw()
    cax_n.figure.canvas.flush_events()   
        
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
    #should return a single probability value of the sensor reading given the ghost's position ~
    # P(O_t | X_t) = P(sensor reading | ghost position)
    # if the ghost is at position s and the sensor reading o is correct (i.e., matches the ghost's position), then the probability is true_prob.
    # if the ghost is at position s and the sensor reading o is incorrect (i.e., does not match the ghost's position), then the probability is (1 - true_prob) 
    # divided by the number of incorrect readings (which is the total number of valid positions minus one).
    # P(O_t | X_t) = true_prob if s corresponds to o, else (1 - true_prob) / (number of valid positions - 1)
    # Hint: you can use the valid_positions list to determine the total number of possible sensor readings and to check if the observation matches
    # the ghost's position.


    pos = valid_positions[s]
    if pos == o:
        return true_prob
    return (1 - true_prob) / (len(valid_positions) - 1)


def bayesian_filter(observation, model):
    #TODO: Implement the Bayesian filter to update the belief vector.
    
    #1. PREDICTION: Compute the belief using transition model below
    # the prediction step should use the transition matrix to compute the predicted belief at time t+1 based on the belief at time t
    # and the ghost's movement model. This can be done using a matrix multiplication between the transition matrix and the belief
    #  vector.
    # predicted_belief = T^T * belief (where T^T is the transpose of the transition matrix)
    # Note: make sure to use the correct dimensions for the matrix multiplication (the transition matrix should be of size (num_states, num_states) 
    # and the belief vector should be of size (num_states)
    predicted_belief = np.dot(model['transition_matrix'].T, model['belief'])

    #2. UPDATE: Incorporate evidence using observation model
    # the update step should use the observation matrix to compute the likelihood of the observation given each possible ghost position, and then multiply this element-wise with the predicted belief to get the updated belief.
    # updated_belief[s] = P(O_t | X_t = s) * predicted_belief[s] for each state s
    # this can be done using an element-wise multiplication between the observation probabilities for the given observation and the predicted belief vector.
    # Note: the observation matrix should be of size (num_observations, num_states) and you should select the row corresponding to the actual observation to get the likelihoods for each state.
    observation_probabilities = model['observation_matrix'][observation]
    updated_belief = observation_probabilities * predicted_belief

    
    #3. NORMALIZATION: Ensure the belief map is a probability distribution
    # after the update step, the belief vector may not sum to 1, so we need to normalize it to ensure it represents a valid probability distribution.
    # this can be done by dividing the updated belief vector by the sum of its elements.
    updated_belief = updated_belief / np.sum(updated_belief)
    model['belief'] = updated_belief

def pacman_reactive_agent_no_random_legal(game_state):
    # Copy of pacman_reactive_agent_no_random, but legal directions use wall perceptions.
    pacman = game_state['pacman']
    opposite_dir = None

    # Determine opposite direction for later use in avoiding reversals.
    if pacman['direction'] == 'up':
        opposite_dir = 'down'
    elif pacman['direction'] == 'down':
        opposite_dir = 'up'
    elif pacman['direction'] == 'left':
        opposite_dir = 'right'
    elif pacman['direction'] == 'right':
        opposite_dir = 'left'

    # food food food
    if pacman_perceptions.dot_up(game_state,2) and not pacman_perceptions.wall_up(game_state):
        game_state['pacman']['direction'] = 'up'
        #print("Moving up towards food")
    elif pacman_perceptions.dot_down(game_state,2) and not pacman_perceptions.wall_down(game_state):
        game_state['pacman']['direction'] = 'down'
        #print("Moving down towards food")
    elif pacman_perceptions.dot_left(game_state,2) and not pacman_perceptions.wall_left(game_state):
        game_state['pacman']['direction'] = 'left'
        #print("Moving left towards food")
    elif pacman_perceptions.dot_right(game_state,2) and not pacman_perceptions.wall_right(game_state):
        game_state['pacman']['direction'] = 'right'
        #print("Moving right towards food")
    # If no ghost or food perceived, choose among legal moves to avoid fixed local loops.
    else:
        legal_dirs = []
        if not pacman_perceptions.wall_up(game_state) and opposite_dir != 'up':
            legal_dirs.append('up')
        if not pacman_perceptions.wall_down(game_state) and opposite_dir != 'down':
            legal_dirs.append('down')
        if not pacman_perceptions.wall_left(game_state) and opposite_dir != 'left':
            legal_dirs.append('left')
        if not pacman_perceptions.wall_right(game_state) and opposite_dir != 'right':
            legal_dirs.append('right')

        if legal_dirs:
            # Prefer to keep moving straight to reduce jitter at intersections.
            if pacman['direction'] in legal_dirs:
                game_state['pacman']['direction'] = pacman['direction']
            else:
                # Otherwise, just pick the first legal direction based on wall perceptions.
                game_state['pacman']['direction'] = legal_dirs[0]
        elif opposite_dir is not None:
            # Dead-end fallback: allow reversing if it is the only move.
            game_state['pacman']['direction'] = opposite_dir
            

def pacman_reactive_agent_no_random_legal_chaseghosts(game_state):
    # Copy of above but now chase ghosts as well 
    # Also adds some range values for food and ghost perception to try to chase them from further away, 
    # which can be tuned for better performance i.e. win vs score.
    pacman = game_state['pacman']
    
    activeghost_detection_range = 3 # How far to check for active (non-frightened) ghosts.
    #frightenedghost_detection_range = 5 # How far to check for frightened ghosts. Set higher than active ghost range to try to chase them from further away.
    food_detection_range = 10 # How far to check for food. Set higher than ghost ranges to try to chase food from further away.
    
    if pacman['direction'] == 'up':
        opposite_dir = 'down'
    elif pacman['direction'] == 'down':
        opposite_dir = 'up'
    elif pacman['direction'] == 'left':
        opposite_dir = 'right'
    elif pacman['direction'] == 'right':
        opposite_dir = 'left'

    if(pacman_perceptions.ghost_frightened(game_state) and not pacman_perceptions.wall_up(game_state)):
        game_state['pacman']['direction'] = 'up'
    elif(pacman_perceptions.ghost_frightened(game_state) and not pacman_perceptions.wall_down(game_state)):
        game_state['pacman']['direction'] = 'down'
    elif(pacman_perceptions.ghost_frightened(game_state) and not pacman_perceptions.wall_left(game_state)):
        game_state['pacman']['direction'] = 'left'
    elif(pacman_perceptions.ghost_frightened(game_state) and not pacman_perceptions.wall_right(game_state)):
        game_state['pacman']['direction'] = 'right'
    else:
        if pacman_perceptions.dot_up(game_state,food_detection_range) and not pacman_perceptions.wall_up(game_state):
            game_state['pacman']['direction'] = 'up'
            #print("Moving up towards food")
        elif pacman_perceptions.dot_down(game_state,food_detection_range) and not pacman_perceptions.wall_down(game_state):
            game_state['pacman']['direction'] = 'down'
            #print("Moving down towards food")
        elif pacman_perceptions.dot_left(game_state,food_detection_range) and not pacman_perceptions.wall_left(game_state):
            game_state['pacman']['direction'] = 'left'
            #print("Moving left towards food")
        elif pacman_perceptions.dot_right(game_state,food_detection_range) and not pacman_perceptions.wall_right(game_state):
            game_state['pacman']['direction'] = 'right'
            #print("Moving right towards food")
        # If no ghost or food perceived, just pick the first legal direction based on wall perceptions.
        else:
            if not pacman_perceptions.wall_up(game_state) and opposite_dir != 'up':
                game_state['pacman']['direction'] = 'up'
            elif not pacman_perceptions.wall_down(game_state) and opposite_dir != 'down':
                game_state['pacman']['direction'] = 'down'
            elif not pacman_perceptions.wall_left(game_state) and opposite_dir != 'left':
                game_state['pacman']['direction'] = 'left'
            elif not pacman_perceptions.wall_right(game_state) and opposite_dir != 'right':
                game_state['pacman']['direction'] = 'right'

def pacmanHMM(game_state):
    if game_state['pacman']['model'] is None:
        model = {}
        model['num_observations'] = model['num_states'] = len(game_state['valid_positions'])

        #TODO: complete the code below to initialise the model. The belief vector is already provided with a uniform prior.
        
        #define the belief vector. To simplify, we will use a 1D vector of size num_states 
        #(i.e., the flattened grid of cells with no obstacles)
        # each entry belief[s] represents the probability that the ghost is in the cell corresponding to state s, given the evidence received so far.
        # the belief should be initialised to a uniform distribution at the start of the game, since we have no information about the ghost's position.
        model['belief'] = np.full(model['num_states'], 1.0/model['num_states']) #1D initial state distribution.

        #1.Define the Transition Matrix (T) - models the probability of moving from one state to any other state
        # for simplicity, we will assume that the ghost can only move to adjacent cells (including diagonals) and that it moves uniformly at random to one
        # of the valid neighbouring cells
        # the transition matrix should be of size (num_states, num_states) where each entry T[s][s'] = P(X_{t+1} = s' | X_t = s) is the probability of 
        # transitioning from state s to state s' according to the ghost's movement model
        transition_matrix = np.zeros((model['num_states'], model['num_states']))
        for s in range(model['num_states']):
            neighbours = get_neighbours(s, game_state['valid_positions'])
            for n in neighbours:
                transition_matrix[s][n] = 1/len(neighbours)
        model['transition_matrix'] = transition_matrix

        
        #2.Define the observation Matrix (E) - models the likelihood of sensor readings given the ghost's position
        # use the calculate_observation_probability function to fill in the observation matrix based on the sensor model and the ghost_true_prob 
        # parameter from the game state
        # the observation matrix should be of size (num_observations, num_states) where each entry E[o][s] = P(O_t = o | X_t = s)   

        observation_matrix = np.zeros((model['num_observations'], model['num_states']))
        for o in range(model['num_observations']):
            for s in range(model['num_states']):
                observation_matrix[o][s] = calculate_observation_probability(game_state['valid_positions'], s, game_state['valid_positions'][o], game_state['pacman']['ghost_true_prob'])
        model['observation_matrix'] = observation_matrix
           
        model['step_count'] = 0

        ##visualisation
        if game_engine.VISUALISE:
            plt.close('all')
            plt.ion()
            fig, ax = plt.subplots()
            cax = ax.imshow(vector_to_matrix(model['belief'], game_state['grid_size'], game_state['valid_positions']), cmap='hot', interpolation='nearest', vmin=0, vmax=1)
            plt.colorbar(cax, label='Probability')
            ax.set_title('Ghost Belief')
            plt.pause(0.001)
            model['fig'] = fig
            model['ax'] = ax
            model['cax'] = cax
            
            '''
            fig_n, ax_n = plt.subplots()
            cax_n = ax_n.imshow(vector_to_matrix(model['belief'], game_state['grid_size'], game_state['valid_positions']), cmap='hot', interpolation='bicubic', vmin=0, vmax=1)
            plt.colorbar(cax_n, label='Probability')
            ax_n.set_title('Ghost Belief in Numbers')
            plt.pause(0.001)
            model['fig_n'] = fig_n
            model['ax_n'] = ax_n
            model['cax_n'] = cax_n
            '''        
            
        game_state['pacman']['model'] = model    

    #get the noisy ghost observation
    observation = game_state['valid_positions'].index(pacman_perceptions.noisy_ghost_position_sensor(game_state, 0, game_state['pacman']['ghost_true_prob']))

    #apply the bayesian filter to update the belief
    bayesian_filter(observation, game_state['pacman']['model'])

    game_state['pacman']['model']['step_count'] += 1

    if game_engine.VISUALISE:
        visualise_belief(vector_to_matrix(game_state['pacman']['model']['belief'], game_state['grid_size'], game_state['valid_positions']), game_state['pacman']['model']['cax'])
    
    if game_engine.VISUALISE and game_state['pacman']['model']['step_count'] == 100:  
        fig_n, ax_n = plt.subplots()
        game_state['pacman']['model']['fig_n'] = fig_n
        game_state['pacman']['model']['ax_n'] = ax_n
    if game_engine.VISUALISE and game_state['pacman']['model']['step_count'] % 100 == 0 and 'ax_n' in game_state['pacman']['model']:    
        visualise_belief_number(vector_to_matrix(game_state['pacman']['model']['belief'],game_state['grid_size'],game_state['valid_positions']),game_state['pacman']['model']['ax_n'])
        
        
    #TODO: Use the updated belief in a reactive agent to move Pac-Man
    #replace this line by your own implementation   
    # for a simple reactive agent, you can compute the most likely position of the ghost based on the belief
    #  (i.e., the state with the highest probability) and then move in the direction that maximizes the distance from that position.
    most_likely_ghost_pos_index = np.argmax(game_state['pacman']['model']['belief'])
    most_likely_ghost_pos = game_state['valid_positions'][most_likely_ghost_pos_index]
    ghost = {'x': most_likely_ghost_pos[0], 'y': most_likely_ghost_pos[1]}
    #game_state['pacman']['ghost_position'] = ghost  

    # get pacman's current position from game-state coordinates
    pacman_pos = (game_state['pacman']['x'], game_state['pacman']['y'])
    # check distance between pacman and the most likely ghost position
    distance = game_engine.manhattan_distance(pacman_pos, most_likely_ghost_pos)
    # if the ghost is far away, move randomly, otherwise move away from the most likely ghost position
    if distance > 5:
        #random_walk(game_state['pacman'], game_state)
        #print("Pacman is far from the most likely ghost position.")
        pacman_reactive_agent_no_random_legal(game_state)
        #pacman_reactive_agent_no_random_legal_chaseghosts(game_state)
    else:
        # Using only perceptions move in the direction that increases the distance from the most likely ghost position. 
        # You cannot use the get_valid_directions or compute_new_pos functions from the game engine.
        # if ghost_frightened move towards ghost instead of away, using the same logic but in reverse (i.e., move in the direction that minimizes the distance to the ghost).
        if pacman_perceptions.ghost_frightened(game_state):
            if most_likely_ghost_pos[0] < pacman_pos[0] and not pacman_perceptions.wall_left(game_state):
                game_state['pacman']['direction'] = 'left'  
            elif most_likely_ghost_pos[0] > pacman_pos[0] and not pacman_perceptions.wall_right(game_state):
                game_state['pacman']['direction'] = 'right'
            elif most_likely_ghost_pos[1] < pacman_pos[1] and not pacman_perceptions.wall_up(game_state):
                game_state['pacman']['direction'] = 'up'
            elif most_likely_ghost_pos[1] > pacman_pos[1] and not pacman_perceptions.wall_down(game_state):
                game_state['pacman']['direction'] = 'down'
            else:
                #random_walk(game_state['pacman'], game_state)
                pacman_reactive_agent_no_random_legal(game_state)
                
        else:
            if most_likely_ghost_pos[0] < pacman_pos[0] and not pacman_perceptions.wall_right(game_state):
                game_state['pacman']['direction'] = 'right'
            elif most_likely_ghost_pos[0] > pacman_pos[0] and not pacman_perceptions.wall_left(game_state):
                game_state['pacman']['direction'] = 'left'
            elif most_likely_ghost_pos[1] < pacman_pos[1] and not pacman_perceptions.wall_down(game_state):
                game_state['pacman']['direction'] = 'down'
            elif most_likely_ghost_pos[1] > pacman_pos[1] and not pacman_perceptions.wall_up(game_state):
                game_state['pacman']['direction'] = 'up'
            else:
                #random_walk(game_state['pacman'], game_state)
                pacman_reactive_agent_no_random_legal(game_state)

        


    #random_walk(game_state['pacman'], game_state)
            
            

    
    
