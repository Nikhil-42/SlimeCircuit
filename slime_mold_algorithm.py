from matplotlib import pyplot as plt
import numpy as np
from numba import prange, njit
import scipy
import scipy.signal
prange = range

SENSOR_ANGLE = np.pi / 4
SENSOR_DISTANCE = 20

@njit
def bilinear(x, y, w, h):
    # Bilinear interpolation
    x += w / 2 - 0.5
    y += h / 2 - 0.5
    
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1
    
    alpha = x - x0
    beta = y - y0
    
    return (
        ((x0, y0), (1 - alpha) * (1 - beta)), 
        ((x1, y0), alpha * (1 - beta)),
        ((x0, y1), (1 - alpha) * beta),
        ((x1, y1), alpha * beta),
    )


# @njit
def sample_field(field, position):
    """Samples the field at the given position, returning the value at that position. 
    A floating point position samples the field using bilinear interpolation.
    
    (0,0) is the center of the field, positive x is to the right, and positive y is up.


    Parameters
    ----------
    field : np.ndarray of shape (w, h)
        The field to sample.
    position : np.ndarray of shape (2,)
        The position to sample the field at.

    Returns
    -------
    float
        The value of the field at the given position.
    """
    value = 0
    for (x, y), weight in bilinear(*position, *field.shape):
        if 0 <= x and x < field.shape[0] and 0 <= y and y < field.shape[1]:
            value += weight * field[x, y]
        else:
            # print('Warning: Out of bounds position', position, (x, y))
            return -float('inf')
    return value


# @njit
def deposit(field, position, amount):
    """Deposits a given amount of pheromones at the given position in the field.
    
    (0,0) is the center of the field, positive x is to the right, and positive y is up.

    Parameters
    ----------
    field : np.ndarray of shape (w, h)
        The field to deposit pheromones in.
    position : np.ndarray of shape (2,)
        The position to deposit the pheromones at.
    amount : float
        The amount of pheromones to deposit.
    """
    for bi in bilinear(*position, *field.shape):
        x, y, weight = bi[0][0], bi[0][1], bi[1]
        field[x, y] += weight * amount


# @njit
def simulate(agents, field, food, num_steps=1):
    """Moves the agents in the simulation one step, updates their direction,
    and deposits pheremones on the field.

    Parameters
    ----------
    agents : np.ndarray of shape (n, 4)
        The agents in the simulation. Each agent has 4 attributes: x, y, dx, dy.
    field: np.ndarray of shape (w, h)
        The field of the simulation containing the pheromone levels.
    food: np.ndarray of shape (m, 2)
        The food in the simulation. Each food has 2 attributes: x, y.
    """
    for _ in range(num_steps):
        debug = np.zeros_like(field)
        for i in prange(len(agents)):
            agent = agents[i]
            # print(agent)

            # Move agent
            agent[:2] += agent[2:]

            # Bounce agent off walls
            x_max = (field.shape[0] - 1) / 2
            y_max = (field.shape[1] - 1) / 2
            if np.abs(agent[0]) > x_max:
                agent[0] -= 2 * (agent[0] - np.sign(agent[0]) * x_max)
                agent[2:] = 0.5 * (np.random.rand(2) - 0.5)
            if np.abs(agent[1]) > y_max:
                agent[1] -= 2 * (agent[1] - np.sign(agent[1]) * y_max)
                agent[2:] = 0.5 * (np.random.rand(2) - 0.5)
            
            
            # Update agent direction
            # Sample the field in front of the agent's position
            position = agent[:2]
            direction = agent[2:] / np.linalg.norm(agent[2:])
            perpendicular = np.array([-direction[1], direction[0]])
            
            front_left_dir = np.cos(-SENSOR_ANGLE) * direction + np.sin(-SENSOR_ANGLE) * perpendicular
            front_dir = direction
            front_right_dir = np.cos(SENSOR_ANGLE) * direction + np.sin(SENSOR_ANGLE) * perpendicular

            front_left_sense = position + SENSOR_DISTANCE * front_left_dir
            front_sense = position + SENSOR_DISTANCE * front_dir
            front_right_sense = position + SENSOR_DISTANCE * front_right_dir
            
            front_left_sample = sample_field(field, front_left_sense)
            front_sample = sample_field(field, front_sense)
            front_right_sample = sample_field(field, front_right_sense)
            
            # Update agent direction
            if front_sample >= front_left_sample and front_sample >= front_right_sample:
                # The front is the best direction
                agent[2:] = front_dir
            elif front_left_sample < front_sample:
                # The front right is the best direction
                agent[2:] = front_right_dir
            elif front_right_sample < front_sample:
                # The front left is the best direction
                agent[2:] = front_left_dir
            else:
                # There is no clear direction, so choose randomly
                agent[2:] = (front_left_dir, front_dir, front_right_dir)[np.random.randint(0, 3)]
                
            # Randomly perturb the direction
            agent[2:] += 0.25 * np.random.randn(2)
                
        # Deposit pheromones
        for i in range(len(agents)):
            agent = agents[i]
            deposit(field, agent[:2], 0.1)
        
        # Deposit food
        for i in range(len(food)):
            deposit(field, food[i], 10)
            
        
        # Gaussian decay of pheromones
        field *= 0.95
        
        # Light mean filtering of pheromones
        field = scipy.signal.convolve2d(field, np.array([[0.5, 1, 0.5], [1, 3, 1], [0.5, 1, 0.5]]) / 9, mode='same')
        
        yield agents, field, debug

if __name__ == '__main__':
    NUM_AGENTS = 500
    NUM_FOOD = 10
    WIDTH = 128 
    HEIGHT = 128
    
    field = np.zeros((WIDTH, HEIGHT))
    agents =  np.empty((NUM_AGENTS, 4))
    agents[:, :2] = (np.random.rand(NUM_AGENTS, 2) - 0.5) * [WIDTH - 1, HEIGHT - 1]
    agents[:, 2:] = 0.5 * (np.random.rand(NUM_AGENTS, 2) - 0.5)

    food = (np.random.rand(NUM_FOOD, 2) - 0.5) * [WIDTH - 1, HEIGHT - 1]
    
    img = None
    for agents, field, debug in simulate(agents, field, food, num_steps=1000):
        if img is None:
            img = plt.imshow(field[:, ::-1].T, cmap='hot', interpolation='bilinear', extent=(-WIDTH/2, WIDTH/2, -HEIGHT/2, HEIGHT/2))
        else:
            img.set_data(field[:, ::-1].T)
        plt.pause(.1)
        plt.draw()