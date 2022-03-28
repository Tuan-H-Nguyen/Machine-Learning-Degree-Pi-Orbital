#%%
import os, sys,copy

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from data_structures.essential_data import *

class Segment:
    """
    Class for segment.
    Attributes:
        + cycles (list): list of cycles that in the segment
        + adjacents (dict): 
    Special method:
        + len: return number of cycles
        + repr: 
        + add (+): add another segment to it and return new segment
            that composed of all cycles in two segment.

    """
    def __init__(self,cycles=None):
        self.cycles = [] if cycles == None else cycles
        self.adjacents = {}
        self.hash_repr = None

    def __len__(self):
        return len(self.cycles)

    def __repr__(self):
        if not self.hash_repr:
            hash_code = sorted([
                cycle.hash_repr for cycle in self.cycles
            ])
            self.hash_repr = hash("".join(hash_code))
        return "Segment("+str(len(self))+","+str(self.hash_repr)+")"

    def __add__(self, segment):
        """
        Add segment to the existing segment to make
        a longer segment.

        Args:
            segment (Segment): _description_

        Returns:
            segment (Segment): segment with the rings of both 
                added segments
        """
        assert isinstance(segment,Segment)
        return self.__class__(self.cycles + segment.cycles)


class FindSegment:
    """
    Search for segments.
    Args:
    + segment_class (Python Class): class of segment.
        Default: Segment
    """
    def __init__(self,segment_class = Segment):
        self.segments = []
        self.visited_ring = []
        self.segment_class = segment_class

    def find_single_segment(
        self,ring,segment=None
        ):
        """
        Depth First Search based algorithm for determining 
        segments. 

        Args:
            ring (_type_): _description_
            segment (_type_, optional): _description_. Defaults to None.
        """
        if ring in self.visited_ring:
            return
        first_flag = False
        if segment == None:
            first_flag = True

        new_segment = self.segment_class([ring]) 
        self.visited_ring.append(ring)

        next_rings = ring.fused_rings.keys()
        if first_flag:
            pass
        else:
            if ring.label == 3 or ring.label == 2 or ring.label == 5:
                self.segments.append(segment + new_segment)
            elif ring.label == 1:
                new_segment = segment + new_segment
            elif ring.label == 0:
                self.segments.append(segment + new_segment)
                return
        for next_ring in next_rings:
            self.find_single_segment(
                next_ring,new_segment)
    
    def find_segments(self,cycles):
        for i,cycle in enumerate(cycles):
            cycle.find_fused_ring(cycles[i:])
            cycle.self_label()
            if len(cycle.fused_rings.keys()) == 1:
                root = cycle
            
        self.find_single_segment(root)
        return self.segments

# %%
