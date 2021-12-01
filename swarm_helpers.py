from operator import mod
import numpy as np
import pyswarms as ps
import pyswarms.backend as P
from pyswarms.backend.swarms import Swarm
import pyswarms.backend as P
from pyswarms.backend.topology import Ring
from itertools import combinations

import implant_helpers as imp

# The error function must accept a tuple of parameter values `params` first, but
# can have other input variables like `features` and `targets`
def err_func(swarm_positions, bounds, model):
    
    # Electrode positions
    scores = []
    for electrode_positions in swarm_positions:
        
        implant = imp.build_implant(electrode_positions)
        n_eff = imp.get_num_effective(implant, model)
        
        # Pairwise distance between electrodes:
        pair_dist = np.array([(e1.x - e2.x) ** 2 + (e1.y - e2.y) ** 2 
                            for (e1, e2) in combinations(implant.electrode_objects, 2)])
        # Maybe try to minimize the average pairwise distance:
        avg_pair_dist = np.mean(pair_dist)
        # Maybe if they're too close to each other, add a penalty:
        too_close_penalty = np.sum(pair_dist <= 200 ** 2)

        # Need to add your own weights, of course, and play with different terms:
        scores.append( -(1000 * n_eff - 0.1 * avg_pair_dist - 1e5 * too_close_penalty))
    
    return np.array(scores)


def particle_swarm(iterations, bounds, model):
    '''
    Particle Swarm optimization loop
    '''
    # Parameters of how particles move
    options = {'c1': 2.3, 'c2': 1.9, 'w': 1.6}
    
    # Initialize the swarm
    swarm = P.create_swarm(n_particles=50, dimensions=120, options=options) # The Swarm Class
    my_topology = Ring() # The Topology Class

    for i in range(iterations):
        # Part 1: Update personal best
        # Computes current cost of electrodes
        swarm.current_cost = err_func(swarm.position, bounds, model=model)
        swarm.pbest_cost = err_func(swarm.pbest_pos, bounds, model=model)  # Compute personal best pos
        # swarm.pbest_pos, swarm.pbest_cost = P.update_pbest(swarm)
        
        if np.min(swarm.current_cost) < swarm.best_cost:
            swarm.best_cost = swarm.current_cost
            swarm.best_pos = swarm.position


        # Part 3: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
        # Takes in the bounds 
        swarm.velocity = my_topology.compute_velocity(swarm)
        swarm.position = my_topology.compute_position(swarm, bounds)
        
        if i % 10 == 0:
            print(f"[Iteration {i}] Best Cost: {swarm.best_cost}")

    return swarm.best_pos, swarm.best_cost 

# Method for dealing with overlapping electrodes

from scipy.spatial.distance import cdist
from scipy.spatial import distance
import random


def findOverlap (ePositions, r):
  coords = ePositions.copy()
  distArray = distance.cdist(coords, coords, 'euclidean')
  arrayLen = distArray.shape[0]
  numColumns = arrayLen - 1
  numRows = arrayLen - 1

  overlapItems = []
  for row in range (0, numRows):
    for col in range (row+1, numColumns + 1):
      arrayItem = distArray[row][col]
      # This line sets what is considered overlap distance
      if arrayItem < (r * 2):
        overlapItems.append(row)
        overlapItems.append(col)
  
  removeIndex = list(set(overlapItems))
  
  positions = []
  for index in removeIndex:
    positions.append(coords[index])
  positions = np.array(positions)

  return positions, removeIndex

def genCoord(lenPoints, bds):
  lb, up = bds
  x = random.sample(range(lb[0], up[1]), lenPoints)
  y = random.sample(range(lb[1], up[1],), lenPoints)

  points = list(zip(x,y))
  points = np.array(points)
  return points
  
def fixOverlap(ePositions, r, bds):
  posArray = ePositions.copy()
  positions, removeIndex = findOverlap(posArray, r)
  newPos = genCoord(len(removeIndex), bds)
 
  for i in range (len(removeIndex)):
    index = removeIndex[i]
    posArray[index] = newPos[i]
  return posArray, len(positions)

# Code to run it in main when I figure out how to optimize it
# ePositions = my_swarm.position.copy()
# ePositions, lenPosition = fixOverlap(ePositions, radius, bounds)
# while (lenPosition > 0):
#     ePositions, lenPosition = fixOverlap(ePositions, radius, bounds)
# my_swarm.position = ePositions