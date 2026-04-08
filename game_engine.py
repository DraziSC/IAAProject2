import pygame
import time
import agents


# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 600, 600
FPS = 0
VISUALISE = True

SCARED_GHOST_TIME = 10 #seconds
PACMAN_CONTINUOUS_MOTION = True
RESPAWN_GHOSTS = True
PACMAN_GHOST_RANGE = 3
PACMAN_DOT_RANGE = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
GOLD = (255, 215, 0)
BLUE = (0, 0, 255)
PINK = (255, 105, 180)

# Game Elements
PACMAN = "P"
GHOST = "G"
DOT = "."
POWER_PELLET = "o"
WALL = "%"
EMPTY = " "

# Load map from file
def load_map_from_file(filename):
    with open(filename, "r") as f:
        map_data = [line.strip() for line in f.readlines()]
        
    return map_data, (len(map_data[0]), len(map_data))

def load_images(cell_size):
    images = {'Blinky':{}}
    images['Blinky']['up'] = [pygame.image.load('images/blinky-up-1.jpg'),pygame.image.load('images/blinky-up-2.jpg')] 
    images['Blinky']['down'] = [pygame.image.load('images/blinky-down-1.jpg'),pygame.image.load('images/blinky-down-2.jpg')]
    images['Blinky']['left'] = [pygame.image.load('images/blinky-left-1.jpg'),pygame.image.load('images/blinky-left-2.jpg')]
    images['Blinky']['right'] = [pygame.image.load('images/blinky-right-1.jpg'),pygame.image.load('images/blinky-right-2.jpg')]
    

    images['Pinky'] = {}
    images['Pinky']['up'] = [pygame.image.load('images/pinky-up-1.jpg'),pygame.image.load('images/pinky-up-2.jpg')]
    images['Pinky']['down'] = [pygame.image.load('images/pinky-down-1.jpg'),pygame.image.load('images/pinky-down-2.jpg')]
    images['Pinky']['left'] = [pygame.image.load('images/pinky-left-1.jpg'),pygame.image.load('images/pinky-left-2.jpg')]
    images['Pinky']['right'] = [pygame.image.load('images/pinky-right-1.jpg'),pygame.image.load('images/pinky-right-2.jpg')]
    
    images['Inky'] = {}
    images['Inky']['up'] = [pygame.image.load('images/inky-up-1.jpg'),pygame.image.load('images/inky-up-2.jpg')]
    images['Inky']['down'] = [pygame.image.load('images/inky-down-1.jpg'),pygame.image.load('images/inky-down-2.jpg')]
    images['Inky']['left'] = [pygame.image.load('images/inky-left-1.jpg'),pygame.image.load('images/inky-left-2.jpg')]
    images['Inky']['right'] = [pygame.image.load('images/inky-right-1.jpg'),pygame.image.load('images/inky-right-2.jpg')]
    
    images['Clyde'] = {}
    images['Clyde']['up'] = [pygame.image.load('images/clyde-up-1.jpg'),pygame.image.load('images/clyde-up-2.jpg')]
    images['Clyde']['down'] = [pygame.image.load('images/clyde-down-1.jpg'),pygame.image.load('images/clyde-down-2.jpg')]
    images['Clyde']['left'] = [pygame.image.load('images/clyde-left-1.jpg'),pygame.image.load('images/clyde-left-2.jpg')]
    images['Clyde']['right'] = [pygame.image.load('images/clyde-right-1.jpg'),pygame.image.load('images/clyde-right-2.jpg')]
    
    images['Frightened'] = {}
    images['Frightened']['blue'] = [pygame.image.load('images/frightened-blue-1.jpg'), pygame.image.load('images/frightened-blue-2.jpg')]
    images['Frightened']['white'] = [pygame.image.load('images/frightened-white-1.jpg'), pygame.image.load('images/frightened-white-2.jpg')]
    
    images['Pacman'] = {}
    images['Pacman']['up'] = [pygame.image.load('images/pacman-up-1.jpg'),pygame.image.load('images/pacman-up-2.jpg')]
    images['Pacman']['down'] = [pygame.image.load('images/pacman-down-1.jpg'),pygame.image.load('images/pacman-down-2.jpg')]
    images['Pacman']['left'] = [pygame.image.load('images/pacman-left-1.jpg'),pygame.image.load('images/pacman-left-2.jpg')]
    images['Pacman']['right'] = [pygame.image.load('images/pacman-right-1.jpg'),pygame.image.load('images/pacman-right-2.jpg')]
    
    
    for name in images:
        for direction in images[name]:
            for i in range(2):
                images[name][direction][i] = pygame.transform.scale(images[name][direction][i], (cell_size, cell_size))
    return images
    
def get_valid_directions(pos, grid, grid_size):
    directions = []
    if pos[0] > 0 and grid[pos[1]][pos[0] - 1] != WALL:
        directions.append('left')
    if pos[0]<grid_size[0]-1 and grid[pos[1]][pos[0] + 1] != WALL:
        directions.append('right')
    if pos[1] > 0 and grid[pos[1] - 1][pos[0]] != WALL:
        directions.append('up')
    if pos[1]<grid_size[1]-1 and grid[pos[1] + 1][pos[0]] != WALL:
        directions.append('down')
    return directions


def compute_new_pos(pos, direction, range=1):
    x, y = pos
    if direction == 'up':
        y -= range
    elif direction == 'down':
        y += range
    elif direction == 'left':
        x -= range
    elif direction == 'right':
        x += range
    return (x, y)

def opposite_direction(direction):
    if direction == 'up':
        return 'down'
    elif direction == 'down':
        return 'up'
    elif direction == 'left':
        return 'right'
    elif direction == 'right':
        return 'left'


def valid_position(new_x, new_y, grid):
    return new_x>= 0 and new_x < len(grid[0]) and new_y>=0 and new_y<len(grid) and grid[new_y][new_x]!=WALL
        


def move_agent(agent, grid):
    new_x, new_y = compute_new_pos((agent['x'], agent['y']), agent['direction'])
    if valid_position(new_x, new_y, grid):
        agent['x'], agent['y'] = new_x, new_y
                            

# Draw the grid
def draw_grid(screen, game_state):
    grid = game_state['grid']
    grid_size = game_state['grid_size']
    cell_size = game_state['cell_size']
    
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            if grid[y][x] == WALL:
                pygame.draw.rect(screen, DARK_GRAY, rect)
            elif grid[y][x] == DOT:
                pygame.draw.circle(screen, GOLD, rect.center, cell_size // 10)
            elif grid[y][x] == POWER_PELLET:
                pygame.draw.circle(screen, GOLD, rect.center, cell_size // 4)

    pacman = game_state['pacman']
    if pacman['image'] is None and pacman['direction'] is None:
        ##for before the first pacman move
        pacman['image'] = game_state['images']['Pacman']['right'][0]

    elif pacman['direction'] is not None:
        if pacman['image'] not in game_state['images']['Pacman'][pacman['direction']]:
            pacman['image'] = game_state['images']['Pacman'][pacman['direction']][0]
        else:
            if pacman['image'] == game_state['images']['Pacman'][pacman['direction']][0]:
                pacman['image'] = game_state['images']['Pacman'][pacman['direction']][1]
            else:
                pacman['image'] = game_state['images']['Pacman'][pacman['direction']][0]

    screen.blit(pacman['image'], (pacman['x']*cell_size, pacman['y']*cell_size))

    for ghost in game_state['ghosts']:
        if ghost['alive']:
            if ghost['direction'] is None:
                if not ghost['scared']:
                    ghost['image'] = game_state['images'][ghost['name']]['right'][0]
                else:
                    ghost['image'] = game_state['images']['Frightened']['blue'][0]

            else:    
                if not ghost['scared']:
                    if ghost['image'] not in game_state['images'][ghost['name']][ghost['direction']]:
                        ghost['image'] = game_state['images'][ghost['name']][ghost['direction']][0]
                    else:
                        if ghost['image'] == game_state['images'][ghost['name']][ghost['direction']][0]:
                            ghost['image'] = game_state['images'][ghost['name']][ghost['direction']][1]
                        else:
                            ghost['image'] = game_state['images'][ghost['name']][ghost['direction']][0]
                else:
                    if ghost['image'] not in [game_state['images']['Frightened']['blue'][0], game_state['images']['Frightened']['blue'][1], game_state['images']['Frightened']['white'][0], game_state['images']['Frightened']['white'][1]]:
                        ghost['image'] = game_state['images']['Frightened']['blue'][0]
                    
                    else:
                        if ghost['image'] == game_state['images']['Frightened']['blue'][0]:
                            ghost['image'] = game_state['images']['Frightened']['blue'][1]
                        elif ghost['image'] == game_state['images']['Frightened']['blue'][1]:
                            ghost['image'] = game_state['images']['Frightened']['white'][0]
                        elif ghost['image'] == game_state['images']['Frightened']['white'][0]:
                            ghost['image'] = game_state['images']['Frightened']['white'][1]                        
                        else:
                            ghost['image'] = game_state['images']['Frightened']['blue'][0]
                    

            screen.blit(ghost['image'], (ghost['x']*cell_size, ghost['y']*cell_size))


def ghost_eaten(ghost, game_state):
    if RESPAWN_GHOSTS:
        ghost['x'], ghost['y'] = ghost['start_x'], ghost['start_y']
        ghost['alive'] = True
    else:
        ghost['alive'] = False
    ghost['scared'] = False
    game_state['score'] += 10
    
def pacman_eaten(game_state):
    game_state['running'] = False

def update_world(game_state):
    grid = game_state['grid']
    pacman_pos = game_state['pacman']['x'], game_state['pacman']['y']
    
    if grid[pacman_pos[1]][pacman_pos[0]] == DOT:
        grid[pacman_pos[1]][pacman_pos[0]] = EMPTY
        check_won(game_state)
        game_state['score'] += 1
    elif grid[pacman_pos[1]][pacman_pos[0]] == POWER_PELLET:
        grid[pacman_pos[1]][pacman_pos[0]] = EMPTY
        check_won(game_state)
        game_state['score'] += 5
        game_state['scared_ghosts_timestamp'] = time.time()
        for ghost in game_state['ghosts']:
            ghost['scared'] = True

    for ghost in game_state['ghosts']:
        if ghost['alive'] and pacman_pos[0] == ghost['x'] and pacman_pos[1] == ghost['y']:
            if ghost['scared']:
                ghost_eaten(ghost, game_state)
            else:
                pacman_eaten(game_state)
                break
    
    if time.time() - game_state['scared_ghosts_timestamp'] > SCARED_GHOST_TIME:
        for ghost in game_state['ghosts']:
            ghost['scared'] = False


    
def check_won(game_state):
    game_state['won'] = True
    game_state['running'] = False
    for row in game_state['grid']:
        if DOT in row or POWER_PELLET in row:
            game_state['won'] = False
            game_state['running'] = True
    
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def check_collisions(pacman, ghosts, game_state):
    for ghost in ghosts:
        if ghost['alive']:
            if ghost['x'] == pacman['x'] and ghost['y'] == pacman['y']:
                if ghost['scared']:
                    ghost_eaten(ghost, game_state)
                else:
                    pacman_eaten(game_state)
                    

# Main function
def main(pacman_policy, ghost_policies, frightened_ghost_policies, map_file = "originalClassic.txt", ghost_true_prob = 0.7):
    map_data, grid_size = load_map_from_file(map_file)
    grid = [list(row) for row in map_data]
    cell_size = WINDOW_WIDTH // (grid_size[0])
    

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Pac-Man")
    clock = pygame.time.Clock()


    # Locate Pac-Man's starting position
    pacman = {'x':1, 'y':1, 'previous_direction': None, 'direction' : None , 'next_direction' : None, 'image':None, 'model':None, 'ghost_true_prob':ghost_true_prob}  # Default position if not found in map
    ghosts = []
    ghost_names = ['Blinky', 'Pinky', 'Inky', 'Clyde']
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            if grid[y][x] == PACMAN:
                pacman['x'] = x
                pacman['y'] = y
                grid[y][x] = EMPTY
            elif grid[y][x] == GHOST:
                ghosts.append( {'start_x':x, 'start_y':y, 'x':x, 'y':y, 'direction':None, 'alive':True, 'image':None, 'name':ghost_names[len(ghosts)], 'scared':False} )
                grid[y][x] = EMPTY

    
    pacman['previous_direction'] = pacman['direction'] = 'right'  # Initial movement direction
    game_state = {'pacman':pacman, 'ghosts':ghosts, 'grid':grid, 'grid_size':grid_size, 'cell_size':cell_size, 'score':0, 'scared_ghosts':0,
                  'running':True, 'scared_ghosts_timestamp':0, 'won':False}
    game_state['images'] = load_images(cell_size)
    
    game_state['valid_positions'] = []
    for i in range(grid_size[0]*grid_size[1]):
        x, y = i % grid_size[0], i // grid_size[0]
        if grid[y][x] != WALL:
            game_state['valid_positions'].append((x,y))
    

    while game_state['running']:
        if not PACMAN_CONTINUOUS_MOTION:
            pacman['direction'] = None
                    
        if pacman_policy != agents.keyboard_controller:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_state['running'] = False
                    
        
        check_collisions(pacman, ghosts, game_state)
        pacman_policy(game_state)        
        move_agent(pacman, game_state['grid'])

        check_collisions(pacman, ghosts, game_state)

        for i, ghost in enumerate(ghosts):
            if ghost['alive']:
                if ghost['scared']:
                    frightened_ghost_policies[i](ghost, game_state)
                else:
                    ghost_policies[i](ghost, game_state)
                move_agent(ghost, game_state['grid'])
            
        update_world(game_state)
        screen.fill(BLACK)
        if VISUALISE:
            draw_grid(screen, game_state)
        pygame.display.flip()
        clock.tick(FPS)

    if game_state['won']:
        print("You won! Score:", game_state['score'])        
    else:
        print("Game Over! Score:", game_state['score'])
    pygame.quit()
    return game_state['score']
    
def set_pacman_direction(game_state,dir, observations, fname):
    pacman = game_state['pacman']
    pacman_pos = (pacman['x'], pacman['y'])
    if dir in get_valid_directions(pacman_pos, game_state['grid'], game_state['grid_size']):
        pacman['previous_direction'] = pacman['direction']
        pacman['direction'] = dir
