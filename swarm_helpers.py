from operator import index, mod
import numpy as np
import pyswarms as ps
import pyswarms.backend as P
from pyswarms.backend.swarms import Swarm
import pyswarms.backend as P
from pyswarms.backend.topology import Ring
from itertools import combinations

from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import distance
import random
import json
import math

import implant_helpers as imp

# The error function must accept a tuple of parameter values `params` first, but
# can have other input variables like `features` and `targets`

def loss_func_basic(swarm_positions, electrode_size, bounds, model):
    
    scores = []
    for electrode_positions in swarm_positions:
        
        implant = imp.build_implant(electrode_positions, electrode_size)
        n_eff = imp.get_num_effective(implant, model)
        scores.append(implant.n_electrodes - n_eff)
          
    return np.array(scores), n_eff

def loss_func_too_close(swarm_positions, electrode_size, bounds, model):
    PENALTY_DISTANCE = electrode_size * 2
    
    # Electrode positions
    scores = []
    for electrode_positions in swarm_positions:
        
        # print('electrode', electrode_positions)
        implant = imp.build_implant(electrode_positions, electrode_size)
        n_eff = imp.get_num_effective(implant, model)
        scores.append(implant.n_electrodes - n_eff)
        
        coords = newXYArray(electrode_positions)
        indexes = distance_penalty(coords, PENALTY_DISTANCE, less_than=True)
        num_too_close = len(indexes)

        score = (implant.n_electrodes - n_eff) + (0.5 * num_too_close)
        # print('score', score)
        scores.append(score)
    
    return np.array(scores), n_eff
  
def loss_func_too_far(swarm_positions, electrode_size, bounds, model):
  PENALTY_DISTANCE = (electrode_size * 2)
  
  # Electrode positions
  scores = []
  for electrode_positions in swarm_positions:
      
      implant = imp.build_implant(electrode_positions, electrode_size)
      n_eff = imp.get_num_effective(implant, model)
      scores.append(implant.n_electrodes - n_eff)
      
      coords = newXYArray(electrode_positions)
      indexes = distance_penalty(coords, PENALTY_DISTANCE, less_than=False)
      num_too_far = len(indexes)

      # print('og', (implant.n_electrodes - n_eff))
      score = (implant.n_electrodes - n_eff) + (0.5 * num_too_far)
      # print('score', score)
      scores.append(score)
  
  return np.array(scores), n_eff

def loss_func_dist_fovea(swarm_positions, electrode_size, bounds, model):
  
  # Electrode positions
  scores = []
  for electrode_positions in swarm_positions:
      
      implant = imp.build_implant(electrode_positions, electrode_size)
      n_eff = imp.get_num_effective(implant, model)
      scores.append(implant.n_electrodes - n_eff)
      
      coords = newXYArray(electrode_positions)
      zero = [(0,0)]
      distArray = distance.cdist(zero, coords)

      sum_dist = np.sum(distArray)

      score = (implant.n_electrodes - n_eff) + (1e-4 * sum_dist)
      # print('score', score)
      scores.append(score)
  
  return np.array(scores), n_eff

def loss_func_convex_hull(swarm_positions, electrode_size, bounds, model):
  # Electrode positions
  scores = []
  for electrode_positions in swarm_positions:
      
      implant = imp.build_implant(electrode_positions, electrode_size)
      n_eff = imp.get_num_effective(implant, model)
      scores.append(implant.n_electrodes - n_eff)
      
      coords = newXYArray(electrode_positions)
      coords = np.array(coords)
      hull = ConvexHull(coords)

      score = (implant.n_electrodes - n_eff) + (0.5e-6 * hull.volume)
      # print('score', score)
      scores.append(score)
  
  return np.array(scores), n_eff


def particle_swarm(num_electrodes, electrode_size, num_particles, iterations, bounds, overlapBounds, model, experiments, loss_func):
    '''
    Particle Swarm optimization loop
    '''
    
    curr_experiment = experiments[num_electrodes][electrode_size][loss_func]

    # Parameters of how particles move
    options = {'c1': 2.3, 'c2': 1.9, 'w': 1.6}
    
    # Initialize the swarm
    swarm = P.create_swarm(n_particles=num_particles, dimensions=int(num_electrodes)*2, options=options) # The Swarm Class
    my_topology = Ring() # The Topology Class

    for i in range(iterations):
        # Update personal best
        swarm.current_cost, n_eff = eval(loss_func + "(swarm.position, int(electrode_size), bounds, model)")
        
        print(swarm.current_cost.shape)
        minIndex = np.argmin(swarm.current_cost)
        
        if (swarm.current_cost[minIndex]) < swarm.best_cost:
          swarm.best_cost = swarm.current_cost[minIndex]
          print(swarm.position.shape, minIndex)
          swarm.best_pos = swarm.position[minIndex]
                  
        # Update position and velocity matrices
        swarm.velocity = my_topology.compute_velocity(swarm)
        swarm.position = my_topology.compute_position(swarm, bounds)

        overlapOverWholeSwarm(swarm.position, overlapBounds)
        
        # if i % 10 == 0:
        #     print(f"[Iteration {i}] Best Cost: {swarm.best_cost}")
        curr_experiment["best_cost_iterations"].append(int(swarm.best_cost))
    
    curr_experiment["best_cost"] = int(swarm.best_cost)
    curr_experiment["best_positions"] = list(swarm.best_pos)
    curr_experiment["best_num_eff"] = int(n_eff)
    
    with open('data.json', 'w') as data:
      json.dump(experiments, data)
      print("Data JSON Updated")
      
    return swarm.best_pos, swarm.best_cost, n_eff 

def distance_penalty(ePositions, penalty_dist, less_than):
  coords = ePositions.copy()
  distArray = distance.cdist(coords, coords, 'euclidean')
  arrayLen = distArray.shape[0]
  numColumns = arrayLen - 1
  numRows = arrayLen - 1

  penalty_items = []
  for row in range (0, numRows):
    for col in range (row+1, numColumns + 1):
      arrayItem = distArray[row][col]
      # This line sets what is considered overlap distance
      if less_than:
        if arrayItem < penalty_dist:
          penalty_items.append(row)
          penalty_items.append(col)
      else:
        # if k closest are not within penalty distance
        if arrayItem > penalty_dist:
          penalty_items.append(row)
          penalty_items.append(col)
  
  penalty_index = list(set(penalty_items))

  return penalty_index


# Method for dealing with overlapping electrodes
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

  return removeIndex

def genCoord(lenPoints, bds):
  lb, up = bds
  x = random.sample(range(lb[0], up[1]), lenPoints)
  y = random.sample(range(lb[1], up[1],), lenPoints)

  points = list(zip(x,y))
  points = np.array(points)
  return points
  
def fixOverlap(ePositions, r, bds):
  posArray = ePositions.copy()
  removeIndex = findOverlap(posArray, r)
  newPos = genCoord(len(removeIndex), bds)
 
  for i in range (len(removeIndex)):
    index = removeIndex[i]
    posArray[index] = newPos[i]
  return posArray, len(removeIndex)

def newXYArray(eArray):
  coords = []
  for i in range(0, len(eArray) -1, 2):
    coords.append((eArray[i], eArray[i+1]))
  return coords

def backTo1DArray (coords):
  eArray = []
  for coord in coords:
    eArray.append(coord[0])
    eArray.append(coord[1])
  return eArray

def overlapOverWholeSwarm(arrayOfPositions, bounds, radius = 100):
  for i in range (len(arrayOfPositions)):
    ePositions = arrayOfPositions[i].copy() 
    coords = newXYArray (ePositions)
    ePositions, lenPosition = fixOverlap(coords, radius, bounds)
    while (lenPosition > 0):
      ePositions, lenPosition = fixOverlap(ePositions, radius, bounds)
    ePositions = backTo1DArray(ePositions)
    arrayOfPositions[i] = ePositions