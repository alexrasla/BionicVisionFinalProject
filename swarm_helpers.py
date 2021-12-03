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
from numpy import linalg

import implant_helpers as imp

# The error function must accept a tuple of parameter values `params` first, but
# can have other input variables like `features` and `targets`
def get_init_num_effective(swarm_positions, electrode_size, bounds, model):

    min_n_eff = math.inf
    for electrode_positions in swarm_positions:
        
        implant = imp.build_implant(electrode_positions, electrode_size)
        n_eff = imp.get_num_effective(implant, model)
        
        if min_n_eff > n_eff:
          min_n_eff = n_eff
          
    return min_n_eff

def loss_func_basic(swarm_positions, electrode_size, bounds, model):
    
    scores = []
    for electrode_positions in swarm_positions:
        
        implant = imp.build_implant(electrode_positions, electrode_size)
        n_eff = imp.get_num_effective(implant, model)
        scores.append(implant.n_electrodes - n_eff)
          
    return np.array(scores)

def loss_func_too_close(swarm_positions, electrode_size, bounds, model):
    PENALTY_DISTANCE = electrode_size * 2
    
    # Electrode positions
    scores = []
    for electrode_positions in swarm_positions:
        
        # print('electrode', electrode_positions)
        implant = imp.build_implant(electrode_positions, electrode_size)
        n_eff = imp.get_num_effective(implant, model)
        
        coords = newXYArray(electrode_positions)
        indexes = distance_penalty(coords, PENALTY_DISTANCE, less_than=True)
        num_too_close = len(indexes)

        score = (implant.n_electrodes - n_eff) + (0.5 * num_too_close)
        scores.append(score)
    
    return np.array(scores)
  
def loss_func_too_far(swarm_positions, electrode_size, bounds, model):
  PENALTY_DISTANCE = (electrode_size * 2)
  
  scores = []
  # Electrode positions
  for electrode_positions in swarm_positions:
      
      implant = imp.build_implant(electrode_positions, electrode_size)
      n_eff = imp.get_num_effective(implant, model)
      
      ePositions = newXYArray(electrode_positions)

      num_too_far = penalityBoundary (ePositions, PENALTY_DISTANCE, bounds)

      # print('og', (implant.n_electrodes - n_eff))
      score = (implant.n_electrodes - n_eff) + (0.5 * num_too_far)
      # print('score', score)
      scores.append(score)
  
  return np.array(scores)

def loss_func_dist_fovea(swarm_positions, electrode_size, bounds, model):
  
  # Electrode positions
  scores = []
  for electrode_positions in swarm_positions:
      
      implant = imp.build_implant(electrode_positions, electrode_size)
      n_eff = imp.get_num_effective(implant, model)
      
      coords = newXYArray(electrode_positions)
      zero = [(0,0)]
      distArray = distance.cdist(zero, coords)

      sum_dist = np.sum(distArray)

      score = (implant.n_electrodes - n_eff) + (1e-4 * sum_dist)
      # print('score', score)
      scores.append(score)
  
  return np.array(scores)

def loss_func_convex_hull(swarm_positions, electrode_size, bounds, model):
  # Electrode positions
  scores = []
  for electrode_positions in swarm_positions:
      
      implant = imp.build_implant(electrode_positions, electrode_size)
      n_eff = imp.get_num_effective(implant, model)
      
      coords = newXYArray(electrode_positions)
      coords = np.array(coords)
      hull = ConvexHull(coords)

      score = (implant.n_electrodes - n_eff) + (0.5e-6 * hull.volume)
      # print('score', score)
      scores.append(score)
  
  return np.array(scores)

def particle_swarm(num_electrodes, electrode_size, num_particles, iterations, bounds, overlapBounds, model, experiments, loss_func):
    '''
    Particle Swarm optimization loop
    '''
    
    curr_experiment = experiments[num_electrodes][electrode_size][loss_func]
    
    # Initialize the swarm
    options = {'c1': 2.3, 'c2': 1.9, 'w': 1.6}    
    swarm = P.create_swarm(n_particles=num_particles, dimensions=int(num_electrodes)*2, options=options) # The Swarm Class
    my_topology = Ring() # The Topology Class
     
    # Initial number effective
    init_num_eff = get_init_num_effective(swarm.position, int(electrode_size), bounds, model)
    curr_experiment["initial_num_effective"] = init_num_eff
    print(f"Initial Number of Effective: {init_num_eff}")   

    for i in range(iterations):
        # Update personal best
        swarm.current_cost = eval(loss_func + "(swarm.position, int(electrode_size), bounds, model)")
        
        minIndex = np.argmin(swarm.current_cost)
        
        if (swarm.current_cost[minIndex]) < swarm.best_cost:
          swarm.best_cost = swarm.current_cost[minIndex]
          swarm.best_pos = swarm.position[minIndex]
          
          #get numnber effective at min index
          implant = imp.build_implant(swarm.best_pos, int(electrode_size))
          best_num_eff = imp.get_num_effective(implant, model)
                  
        # Update position and velocity matrices
        swarm.velocity = my_topology.compute_velocity(swarm)
        swarm.position = my_topology.compute_position(swarm, bounds)

        overlapOverWholeSwarm(swarm.position, overlapBounds)
        
        curr_experiment["best_cost_iterations"].append(int(swarm.best_cost))
    
    curr_experiment["best_cost"] = int(swarm.best_cost)
    curr_experiment["best_positions"] = list(swarm.best_pos)
    curr_experiment["best_num_eff"] = int(best_num_eff)
    
    with open('data.json', 'w') as data:
      json.dump(experiments, data)
      print("Data JSON Updated")
      
    return swarm.best_pos, swarm.best_cost, best_num_eff 

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
      if arrayItem < penalty_dist:
        penalty_items.append(row)
        penalty_items.append(col)
  
  penalty_index = list(set(penalty_items))

  return penalty_index


def penalityBoundary (ePositions, penalty_dist, bds):
  x, y = bds
  coords = newXYArray(ePositions)
  boundaryPoints = []
  # Need to calculate the points of all 4 corners and make lines for the boundaries 
  bottomLeft = (x[0], y[0])
  topLeft = (x[0], y[1])
  bottomRight = (x[1], y[0])
  topRight = (x[1], y[1])

  leftLine = (bottomLeft, topLeft)
  rightLine = (bottomRight, topRight)
  bottomLine = (bottomLeft, bottomRight)
  topLine = (topLeft, topRight)

  lines = [leftLine, rightLine, bottomLine, topLine]

  boundaryPoints = []

  for i in range(len(ePositions)):
    for line in lines:
      point1 = np.array(line[0])
      point2 = np.array(line[1])
      point3 = np.array(ePositions[i])
      distToBoundary = np.abs(np.cross(point2 - point1, point1 - point3))
      if distToBoundary < penalty_dist:
        boundaryPoints.append(i)
  
  tooCloseList = list(set(boundaryPoints))
  return len(tooCloseList)


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