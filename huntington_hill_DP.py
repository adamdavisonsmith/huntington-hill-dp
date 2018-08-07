#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:58:21 2018

@author: ads22
"""

import math
import numpy as np
import scipy.stats 
import warnings
# import heapq as hq
# import pandas as pd

# The following function are useful helpers.

def second_argmax(x):
    ### x: numeric list (or array) of nonzero length
    ### Returns: indices of second highest and maximum elements, 
    ### where index of second highest is -1 if array has length 1.
    second_val = -np.inf
    second_argmax = -1
    max_val = x[0]
    argmax = 0
    for i in range(1,len(x)):
        if (x[i] > max_val):
            second_argmax = argmax
            second_val = max_val
            argmax = i 
            max_val = x[i]
        elif (x[i] > second_val):
            second_argmax = i
            second_val = x[i]
    return second_argmax, argmax

def second_argmin(x):
    ### x: numeric numpy array of nonzero length
    return second_argmax(- x)



def huntington_hill(populations_array,num_seats):
    ### populations is a numpy array of populations
    ### num_seats is the desired number of seats to assign
    num_states = len(populations_array)
    if (num_states > num_seats):
        print("More states than seats!")
        return None
    representatives = np.ones(num_states)
    priorities = populations_array / math.sqrt(2)
    last_state = None
    last_priority = None
    second_last_state = None
    second_last_state_priority = None
    for j in range(num_states,num_seats):
        highest = np.argmax(priorities) # index of state with highest priority. It will receive this seat
        representatives[highest] +=  1    
        # Update this state's priority
        priorities[highest] = populations_array[highest]/math.sqrt(
            representatives[highest] * (representatives[highest]+1))
    # After the for loop,
    # representatives contains the number of reps assigned to each state.
    # "priorities" is an array of the priority of each state fo the next (unassigned) seat. 
    # invariant: priorities = populations_array / np.sqrt(representatives * (representatives + 1))
    
    
    ############
    # Computing the distance to instability. 
    ############
    # CLAIM: the minimal change consists of 
    # either adding or removing people to/from a single state. 
    # (Proof omitted here.)
    ############

    # We will need the highest and second-highest of the current priorities
    second_high, high = second_argmax(priorities)
    
    # We also need the lowest and second-lowest of the priorities that each state had 
    # when it was last assigned a seat

    prev_norm_factors = np.sqrt(representatives*(representatives - 1)) 
    #prev_priorities = populations_array / prev_norm_factors
    prev_priorities = np.zeros(num_states)
    for i in range(num_states):
        if (representatives[i] == 1):
            prev_priorities[i] = np.inf
        else: # normalization factor is not 0
            prev_priorities[i] = populations_array[i] / prev_norm_factors[i]

    second_low, low = second_argmin (prev_priorities)
        
    ############
    # Step 1: Compute the minimal population addition which would gain some state a new seat.
    # For all but one state, this is the addition of population needed to raise their 
    # priority over min(prev_priorities)
    # Let's call this its "normalized deficiency"
    normalization_factors = np.sqrt(representatives*(representatives + 1))
    normalized_defs = (prev_priorities[low] - priorities) * normalization_factors
    # Need to fix computation of normalized gap for last state. 
    # In order for it to gain a seat, it's current priority would have to move above the second 
    # highest of the priorities. 
    normalized_defs[low] = (prev_priorities[second_low] - 
                                       priorities[low]) * normalization_factors[low]

    min_addition = np.min(normalized_defs)
    min_addition_index = np.argmin(normalized_defs)

    ############
    # Step 2: Compute the number of people we would have to drop from a state to cost it a seat.
    # For all but one state, this is the number of people we need to drop to bring 
    # the state's priority at the time its last seat was assigned down to the priority 
    # of the state which would get the next seat, if there were one. 
    # We'll call this a normalized surplus

    # The natural way to write this would be the following:    
    #    normalized_surplus = (prev_priorities - priorities[high]) * prev_norm_factors 
    # but Python doesn't like arrays with np.inf values; instead, we write:
    normalized_surplus = populations_array - ( priorities[high] * prev_norm_factors )

    # Now we need to fix the surplus for the state which would have gotten the next seat. 
    normalized_surplus[high] = populations_array[high] - priorities[second_high] * prev_norm_factors[high]

    
    min_removal = np.min(normalized_surplus)
    min_removal_index = np.argmin(normalized_surplus)
        
    distance_to_instability = np.ceil(min(min_addition, min_removal))
        
    stats = dict([
        ('representatives', representatives),
        ('distance to instability', distance_to_instability),
        ('priorities', priorities),
        ('previous priorities', prev_priorities),
        ('min addition', min_addition),
        ('min add index', min_addition_index),
        ('min removal', min_removal),
        ('min rem index', min_removal_index),
        ('defs', normalized_defs),
        ('surps', normalized_surplus)
        ])
    
    return stats


def laplace_histogram(populations_array, epsilon):
    ### Returns a vector of populations with  Laplace noise added
    num_states = len(populations_array)
    return populations_array + np.random.laplace(scale = 1/epsilon, size = (num_states))

def geometric_histogram(populations_array, epsilon):
    ### Returns a vector of populations with discrete Laplace noise added
    num_states = len(populations_array)
    noise = scipy.stats.dlaplace.rvs(epsilon, size = num_states)
    return populations_array + noise

def hh_robustness_to_noise_of_instance(populations_array, num_seats, epsilon, num_reps):
    ### This function repeats a noise-tolerance experiment several times: 
    ### Noise is added to state populations, and the function tracks how
    ### often the apportionment changes (and by how much, on average)
    ### populations_array is a numpy array  containing the states' populations
    ### num_seats is the desired number of seats to assign
    ### epsilon is the parameter used for DP noise addition (Laplace noise)
    ### num_reps is the number of attempts that are made
    ### Returns the number of changed outputs, the average change, 
    ### and the instance's distance to instability.
    ref_stats = huntington_hill(populations_array, num_seats)
    ref_reps = ref_stats['representatives']
    distance_to_instab = ref_stats['distance to instability']
    changed_outputs = 0
    total_changes = 0
    for i in range(num_reps):
        noisy_pops = geometric_histogram(populations_array, epsilon)
        reps = huntington_hill(noisy_pops, num_seats)['representatives']
        if not np.array_equal(reps, ref_reps):
            changed_outputs = changed_outputs + 1
        total_changes = total_changes + np.sum(np.abs(reps - ref_reps))
    if (changed_outputs == 0):
        avg_change = 0.0
    else:
        avg_change = total_changes / changed_outputs
    return changed_outputs, avg_change, distance_to_instab




def huntington_hill_projection(reported_populations, target_apportionment, norm=1):
    ### Given reported populations and a target apportionment (both numpy arrays 
    ### of the same length), returns a nearest vector of populations that is consistent
    ### with the target. Nearest is measured according to the specified
    ### nrom (currently only l1 is supported).
    
    # renaming for compactness and readability
    pops = reported_populations
    targs = target_apportionment
    
    u_divisors = np.sqrt(targs * (targs - 1)) # divisor for upper bound on T
    l_divisors = np.sqrt(targs * (targs + 1)) # divisor for lower bound on T
    # Note that true threshold for these pops would satisfy
    #   pops/l_divisors <= T <= pops/u_divisors
    # We want to find the minimal modification to pops that makes these hold.
    
    
    def min_delta_thresh(T):
        ### Given a threshold T (a float), this returns an optimal 
        ### vector Delta that minimizes change to the populations
        ### that would make HH return the target apportionment using 
        ### this particular threshhold
        # Can probably rewrite the following code more compactly
        # using array notation
        delta = np.zeros(len(pops))
        for i in range(len(pops)):
            if (T * l_divisors[i] < pops[i]): 
                # equivalently, T < pops[i] / l_divisors[i]
                # Need to reduce the population
                #  delta[i] will be negative
                delta[i] = np.floor(pops[i] - T * l_divisors[i] )
            elif (T * u_divisors[i] > pops[i]): 
                # equivalently, T > pops[i] / u_divisors[i]
                # Need to increase population
                #     delta[i] will be negative
                # Note that this elif test does the right thing when
                #   u_divisors[i] = 0, i.e. targs[i]=1,
                # namely, it passes and seets delta[i]=0
                delta[i] = np.ceil(pops[i] - T * u_divisors[i])
            else:
                delta[i] = 0 # redundant but included for clarity
#        for i in range(len(pops)):
#            if (delta_plus[i] > 0):
#                delta[i] = delta_plus[i]
#            elif (delta_minus[i] < 0)
#                delta[i] = delta_minus[i]
#            else:
        return delta
    
    
    # Now we need to find the best threshold
    
    # The following straightforward approach yields a quadratic time algorithm 
    # One can get running time O(n log n) where n is the number of states
    # by using the fact, not proven here, that for any Lp norm, the function
    # that maps t to (Lp distance)^p to the nearest data set with for which 
    # t works as a threshold for the target number of representatives is
    # itself convex (it is a sum of convex functions). So the best thresholds
    # will be contiguous, and the distance function will increase as you move 
    # away from those thresholds in either direction.
    
    # For l1 minimization, we know optimal threshold, WLOG, will be 
    # one of the points where the slope of the objective function
    # changes, i.e. a point in b or a point in d

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # This wrapper drops the "division by zero" warnings
        # generated for states with one target representative
        b = pops / l_divisors
        d = pops / u_divisors
        possible_thresh = np.concatenate((b, d))
        possible_thresh.sort()
        
    best_delta = None
    best_t = None
    best_dist = np.infty
    dist_by_thresh = []
    thresholds_used = []
    for t in possible_thresh:
        if (t < np.infty):
            delta = min_delta_thresh(t)
            dist = np.sum(np.abs(delta)) #Change this line to change norms
            # Use dist = np.linalg.norm(delta, ord = p) to get L_p-norm
            thresholds_used.append(t)
            dist_by_thresh.append(dist)
            if dist < best_dist:
                best_dist = dist
                best_t = t
                best_delta = delta
            
    
    output = {}
    output['Projected populations'] = pops + best_delta
    output['Delta'] = best_delta
    output['Threshold'] = best_t
    output['Distance'] = best_dist
    output['All Thresholds'] = np.array(thresholds_used)
    output['All Distances'] = np.array(dist_by_thresh)
    return output

def hh_projection_after_noise_on_instance(populations_array, num_seats, epsilon, num_reps):
    ### This function repeats a noise-tolerance experiment several times: 
    ### Noise is added to state populations. The function tracks how often the 
    ### apportionment changes, how much it changes, and how far the noisy 
    ### populations are from the nearest pops with the true apportionment
    ### populations_array is a numpy array  containing the states' populations
    ### num_seats is the desired number of seats to assign
    ### epsilon is the parameter used for DP noise addition (Laplace noise)
    ### num_reps is the number of attempts that are made
    ### Returns a dictionary with miscellaneous results (sorry...)
    ref_stats = huntington_hill(populations_array, num_seats)
    ref_reps = ref_stats['representatives']
    distance_to_instab = ref_stats['distance to instability']
    changed_outputs = 0
    total_changes = 0
    total_distances = 0 
    for i in range(num_reps):
        noisy_pops = geometric_histogram(populations_array, epsilon)
        reps = huntington_hill(noisy_pops, num_seats)['representatives']
        if not np.array_equal(reps, ref_reps):
            changed_outputs = changed_outputs + 1
            projection_result = huntington_hill_projection(noisy_pops, ref_reps)
            total_changes = total_changes + np.sum(np.abs(reps - ref_reps))
            total_distances = total_distances + projection_result['Distance']
    if (changed_outputs == 0):
        avg_change = 0.0
        avg_distance = 0.0
    else:
        avg_change = total_changes / changed_outputs
        avg_distance = total_distances / changed_outputs 
    return changed_outputs, avg_change, avg_distance, distance_to_instab

    
    
            

    
    
    
    
    
    
    
    
        
        
        
    
    
    