import game_engine
import random

def dot_up(game_state, range=1):
    pacman = game_state['pacman']
    
    visibility = 0
    while(visibility < range):
        ##check how far we can see. This is limited by both range and walls
        if pacman['y'] - visibility < 0 or game_state['grid'][pacman['y'] - visibility][pacman['x']] == game_engine.WALL:
            break
        else:
            if game_state['grid'][pacman['y']-visibility][pacman['x']] == game_engine.DOT or game_state['grid'][pacman['y']-visibility][pacman['x']] == game_engine.POWER_PELLET:
                return True
            visibility += 1
    return False
    
def dot_down(game_state, range=1):
    pacman = game_state['pacman']
    
    visibility = 0
    while(visibility < range):
        ##check how far we can see. This is limited by both range and walls
        if pacman['y'] + visibility >= game_state['grid_size'][1] or game_state['grid'][pacman['y'] + visibility][pacman['x']] == game_engine.WALL:
            break
        else:
            if game_state['grid'][pacman['y']+visibility][pacman['x']] == game_engine.DOT or game_state['grid'][pacman['y']+visibility][pacman['x']] == game_engine.POWER_PELLET:
                return True
            visibility += 1
    return False


    
def dot_left(game_state, range=1):
    pacman = game_state['pacman']
    
    visibility = 0
    while(visibility < range):
        ##check how far we can see. This is limited by both range and walls
        if pacman['x'] - visibility < 0 or game_state['grid'][pacman['y']][pacman['x'] - visibility] == game_engine.WALL:
            break
        else:
            if game_state['grid'][pacman['y']][pacman['x'] - visibility] == game_engine.DOT or game_state['grid'][pacman['y']][pacman['x'] - visibility] == game_engine.POWER_PELLET:
                return True
            visibility += 1
    return False
    
    
def dot_right(game_state, range=1):
    pacman = game_state['pacman']
    
    visibility = 0
    while(visibility < range):
        ##check how far we can see. This is limited by both range and walls
        if pacman['x'] + visibility >= game_state['grid_size'][0] or game_state['grid'][pacman['y']][pacman['x'] + visibility] == game_engine.WALL:
            break
        else:
            if game_state['grid'][pacman['y']][pacman['x'] + visibility] == game_engine.DOT or game_state['grid'][pacman['y']][pacman['x'] + visibility] == game_engine.POWER_PELLET:
                return True
            visibility += 1
    return False

def ghost_frightened(game_state):
    #returns true if at least one ghost is frightened

    for ghost in game_state['ghosts']:
        if ghost['alive'] and ghost['scared']:
            return True
    return False

def wall_up(game_state):
    pacman = game_state['pacman']
    
    if pacman['y'] == 0 or game_state['grid'][pacman['y'] - 1][pacman['x']] == game_engine.WALL:
        return True
    return False

def wall_down(game_state):
    pacman = game_state['pacman']
    
    if pacman['y'] == game_state['grid_size'][1]-1 or game_state['grid'][pacman['y'] + 1][pacman['x']] == game_engine.WALL:
        return True
    return False

def wall_left(game_state):
    pacman = game_state['pacman']
    
    if pacman['x'] == 0 or game_state['grid'][pacman['y']][pacman['x'] - 1] == game_engine.WALL:
        return True
    return False

def wall_right(game_state):
    pacman = game_state['pacman']
    
    if pacman['x'] == game_state['grid_size'][0]-1 or game_state['grid'][pacman['y']][pacman['x'] + 1] == game_engine.WALL:
        return True
    return False

def noisy_sensor(percept, true_prob = (0.9,0.1)):
    #true_prob = (prob of true|true, prob of true|false)    
    if percept:
        return random.random() < true_prob[0]
    else:
        return random.random() < true_prob[1]

def noisy_ghost_position_sensor(game_state, ind_ghost, prob):
    #the sensor returns the accurate position of the ghost with probability prob
    #and a random valid position otherwise (excludes cells with walls and the ghost's current position)
    if random.random()<prob:
        return (game_state['ghosts'][ind_ghost]['x'], game_state['ghosts'][ind_ghost]['y'])
    else:
        candidates = game_state['valid_positions'][:]
        candidates.remove((game_state['ghosts'][ind_ghost]['x'], game_state['ghosts'][ind_ghost]['y']))
        return random.choice(candidates)
    

def pacman_distance_to_ghost(game_state, most_likely_ghost_pos):
    pacman = game_state['pacman']
    
    pacman_pos = (game_state['pacman']['x'], game_state['pacman']['y'])

    # check distance between pacman and the most likely ghost position
    distance = game_engine.manhattan_distance(pacman_pos, most_likely_ghost_pos)

    return distance
