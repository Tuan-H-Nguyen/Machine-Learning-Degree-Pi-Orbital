#%%
import os, sys, itertools
import numpy as np 

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from data_structures.essential_data import *
from data_structures.molecular_graph import MolecularGraph
from data_structures.cycles import Cycle

from poly_rings.segments import FindSegment, Segment
from poly_rings.orientation import update_segments_orientation

from utils.logic import intersection

def rotate(_list):
    return [_list[-1]] + _list[0:-1]

class FusedRing(Cycle):
    def __init__(self, atoms):
        """
        Attributes:
        + fused_rings (dict): dictionary of 
            {adjacent fused ring: fused/overlaped atoms} 
        + label: 
        """
        super().__init__(atoms)
        self.fused_rings = {}
        self.label = None

    def find_fused_ring(self,rings):
        """
        Update a ring or a list of rings with a dictionary 
        that its keys are rings that fused to the inquired
        ring(s) and values are corresponding pairs of fused 
        atoms (atoms between the pair of rings). 

        Args:
            rings (Ring or list of Rings): 
        """
        if isinstance(rings,FusedRing) and rings == self:
            pass
        elif isinstance(rings,FusedRing):
            fused_atoms = intersection(rings.atoms,self.atoms)
            if fused_atoms:
                self.fused_rings[rings] = fused_atoms
                rings.fused_rings[self] = fused_atoms
        elif isinstance(rings,list):
            for ring in rings:
                self.find_fused_ring(ring)

    def self_label(self):
        """
        Label rings. 
        5 for thiophene rings
        0, 3 for rings that fused with 1 or 3 rings, respectively.
        1 for ring that fused with 2 rings if 3 rings stays in 
            a line.
        otherwise 2 
        
        Returns:
        + string: 1 - 5
        """
        if len(self.atoms) == 5:
            self.label = 5
        elif len(self.fused_rings.keys()) == 1:
            self.label = 0
        elif len(self.fused_rings.keys()) == 3:
            self.label = 3
        elif len(self.fused_rings.keys()) == 2:
            ring = self.atoms
            fused_atoms = list(
                itertools.chain(*self.fused_rings.values()))
            while ring[0] not in fused_atoms or ring[1] not in fused_atoms:
                ring = rotate(ring) 
            if ring[3] in fused_atoms and ring[4] in fused_atoms:
                self.label = 1
            else: self.label = 2
            
    def __repr__(self):
        return "FusedRing("+str(len(self))+","+self.hash_repr+")"

class PolyRingGraph(MolecularGraph):
    """
    Class for molecular graph that represent the 
    PAHs and Thienoacenes. 
    New addition is the object is initiated with SMILES
    string, run cycle and segments search, segments adjacency 
    search within the initiation.

    Init Args:
    + smiles (String): SMILES string of
        PAH or thienoacenes.
    """
    def __init__(self,smiles):
        super().__init__()
        self.from_smiles(smiles)
        self.find_cycles()
        self.cycles = [
            FusedRing(ring.atoms) 
            for ring in self.cycles
        ]
        fs = FindSegment()
        self.segments = fs.find_segments(self.cycles)
        update_segments_orientation(self)
#%%
"""

#sample = "s1ccc2c1c1ccccc1c1cc3cc4c(cc3cc21)c1c(cccc1)c1c4ccs1"
#sample = "C12=CC=CC=C1C=C3C(C=CC4=C3C5=C(C=C(C=CC=C6)C6=C5)C7=C4C=C(C=CC=C8)C8=C7)=C2"
sample = "C12=C(C=CC=C3)C3=C(C=C(C=CC4=C5C=C6C(C=CC=C6)=C4)C5=C7)C7=C1C=CC=C2"
#sample = "C1(C=CC=C2)=C2C=C(C(C=CC3=C4C=CC=C3)=C4C5=C6C=C7C(C(C=CC=C8)=C8C9=C7C=CC=C9)=C5)C6=C1"
#sample = "c(sc1c2)(cc(c(c3)c4)cc5c4cccc5)c3c1cc6c2cccc6"
#sample = "s1ccc2c1c1ccccc1c1cc3cc4c(cc3cc21)c1c(cccc1)c1c4ccs1"
graph = PolyRingGraph(sample)
for seg in graph.segments:
    print("###################")
    
    (seg)
    print(seg.cycles)
"""

# %%
