import numpy as np
import os, sys

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from data_structures.essential_data import PERIODIC_TABLE
from data_structures.cycles import CycleFinding, Cycle
"""
For separate compound rings.
The result of the cycle finding algorithm most of the time 
contain bigger rings that compose of two or more rings.
This script is dedicated to solve this problem by brute force.

"""

def decompose_rings(ring):
    if isinstance(ring,Cycle):
        ring = ring.atoms
    cf = CycleFinding(allowed_nodes=ring)
    smaller_cycles = cf.find_cycle(start_node=ring[0])
    return smaller_cycles

def sort_cycle(cycles_list):
    single_cycles = []
    multi_cycles = []
    for cycle in cycles_list:
        if len(decompose_rings(cycle)) > 1:
            multi_cycles.append(cycle)
        else:
            single_cycles.append(cycle)
    return single_cycles, multi_cycles

def extract_rings(
    ring,forbidden_edge
    ):
    assert isinstance(ring, Cycle)
    ring = ring.atoms
    cf = CycleFinding(
        allowed_nodes = ring
    )
    e1,e2 = forbidden_edge 
    cf.forbidden_edge = {
        e1:[e2], e2:[e1]
    }
    rings = cf.find_cycle(ring[0])
    s_list, m_list = sort_cycle(rings) 
    return s_list, m_list

class BruteSearchRing:
    def __init__(self,single_ring):
        assert isinstance(single_ring,list)
        self.single_ring = single_ring
        self.searched_ring = []
        self.to_search_ring = []

    def is_single_ring(self,ring):
        _list = [ring.hash_repr for ring in self.single_ring]
        return ring.hash_repr in _list
        
    def is_searched_ring(self,ring):
        _list = [ring.hash_repr for ring in self.searched_ring]
        return ring.hash_repr in _list
    
    def is_to_search_ring(self,ring):
        _list = [ring.hash_repr for ring in self.to_search_ring]
        return ring.hash_repr in _list

    def search(self,ring):
        #skip if the compound ring has been searched
        #this skip also remove this ring from the 
        # to-searched list in the loop below
        if self.is_searched_ring(ring):
            return
        #all ring go through the check point is searched
        #, hence, it is added to the searched list
        self.searched_ring.append(ring)
        for i in range(-1,len(ring)-1):
            e1,e2 = ring[i], ring[i+1]
            s_list,m_list  = extract_rings(ring,(e1,e2))
            for r in s_list:
                if not self.is_single_ring(r):
                    self.single_ring.append(r)
            for r in m_list:
                if not self.is_to_search_ring(r):
                    self.to_search_ring.append(r)
    
    def brute_search(self,ring):
        self.to_search_ring.append(ring)
        while self.to_search_ring:
            ring = self.to_search_ring[0]
            self.to_search_ring.pop(0)
            self.search(ring)


