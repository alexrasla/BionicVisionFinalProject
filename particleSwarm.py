import numpy as np
import pyswarms as ps
import pyswarms.backend as P
from pyswarms.backend.swarms import Swarm
import pyswarms.backend as P
from pyswarms.backend.topology import Ring

import implantHelpers as helpers

def createSwarm(implant):
    '''
    Creates particle swarm from and implant
    '''
    # Create initial positions (from implant) and veloctities (random)
    init_positions = []
    for name, electrode in implant.electrodes.items():
        init_positions.append((electrode.x, electrode.y))

    init_positions = np.array(init_positions)
    init_velocities = P.generate_velocity(n_particles=implant.n_electrodes, dimensions=2)

    # Parameters of how particles move
    options = {'c1': 2.3, 'c2': 1.9, 'w': 1.6}
    
    # Initialize the swarm
    swarm = Swarm(position=init_positions, velocity=init_velocities, options=options)
    
    return swarm

def err_func(targets, features):
    '''
    Targets is number of actual electrodes
    Features is number of effective electrodes
    '''
    return (targets - features)

def ps_optimization(swarm, iterations, bounds, model, electrode_radius):
    '''
    Particle Swarm optimization loop
    '''
    
    np.seterr(invalid='ignore')
    my_topology = Ring() # The Topology Class

    for i in range(iterations):
        # Part 1: Update personal best
        implant = helpers.buildElectrodeArray(swarm.position, electrode_radius)
        effective = helpers.numberOfEffectiveElectrodes(implant, model)
    
        # Computes current cost of electrodes
        swarm.current_cost = err_func(implant.n_electrodes, effective)

        if swarm.current_cost < swarm.best_cost:
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