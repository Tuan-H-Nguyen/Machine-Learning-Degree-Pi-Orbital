#%%
import numpy as np
import os, sys, itertools

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from data_structures.essential_data import *
from data_structures.cycles import CycleFinding

from poly_rings.segments import Segment

from utils.logic import intersection

def overlap(segment1,segment2):
    """
    Check if two segments have one ring in common. If yes, 
    their adjacent dictionary are updated with each other 
    under '120' key, if not, nothing happens.

    Args:
        segment1 (Segment): 
        segment2 (Segment): 

    Returns: True 
    """
    if isinstance(segment2,Segment):
        inter = intersection(segment1.cycles,segment2.cycles)
        assert len(inter) <= 1
        if inter and len(inter[0]) != 5:
            # not considering pair of segments that overlap at thiophene ring 
            # is adjacent to each other since once of them will be neglected 
            # anyway. 
            segment1.adjacents["120"].append(segment2)
            segment2.adjacents["120"].append(segment1)
    elif isinstance(segment2,list):
        for seg in segment2:
            overlap(segment1,seg)
    return True

def check_orientation(seg1,seg2):
    """Find orientation of two segments

    Args:
        seg1 (Segment): 
        seg2 (Segment): 

    Returns: either
    + None: if two segments are found to either be the same
        or have one ring in common or not having one adjacent
        segment (segment forms with them an angle of 120) 
        in common
    OR
    + String: "60" and "0" for angle of 60 and parallel, 
        respectively
    """
    if seg1 == seg2:
        #if two segments are found to be the same
        return None
    
    seg1_adj = seg1.adjacents["120"]
    seg2_adj = seg2.adjacents["120"]
    
    seg3 = intersection(seg1_adj,seg2_adj)

    if seg1 not in seg2_adj and seg2 not in seg1_adj and seg3:
        pass
    else:
        """
        have one ring in common (effectively form angle of 120,
        thus is out of question)
        or 
        not having one adjacent segment (segment forms with 
        them an angle of 120) in common
        """
        return None
    
    assert len(seg3) == 1
    seg3 = seg3[0]

    fused_atom_pairs = []

    # find the ring that is the overlap betwwen each of the
    # segment in the pair (1,2) and the segment between 
    # them (3)
    # then => pairs of atoms (form w=edge within seg 1) that 
    # are where segment 1 and 2 attach to segment 3
    seg1_3_overlap = intersection(seg1.cycles,seg3.cycles)
    assert len(seg1_3_overlap) == 1
    seg1_3_overlap = seg1_3_overlap[0]

    for ring,pair in seg1_3_overlap.fused_rings.items():
        if ring in seg1.cycles:
            fused_atom_pairs.append(pair)

    seg2_3_overlap = intersection(seg2.cycles,seg3.cycles)
    assert len(seg2_3_overlap) == 1
    seg2_3_overlap = seg2_3_overlap[0]

    for ring,pair in seg2_3_overlap.fused_rings.items():
        if ring in seg2.cycles:
            fused_atom_pairs.append(pair)

    # edges to ignore when run the cycle detecting algorithm
    # they are edges shared between pairs of rings 
    forbidden_edge = []
    for i,cycle in enumerate(seg3.cycles[1:]):
        for ring ,atom_pair in cycle.fused_rings.items():
            if ring == seg3.cycles[i]:
                forbidden_edge.append(atom_pair)

    # initiate cycle finding algorithm for segment in between (3)
    cf = CycleFinding(
        allowed_nodes=list(
            itertools.chain(*[cycle.atoms for cycle in seg3.cycles]))
        )

    # add forbidden/neglected edges to the algorithm
    for edge in forbidden_edge:
        atom1, atom2 = edge[0],edge[1]
        cf.forbidden_edge[atom1] = [atom2]
        cf.forbidden_edge[atom2] = [atom1]

    # run to find the big ring encompass the whole segment with
    # no smaller ring within.
    stitched_cycle = cf.find_cycle(
        seg3.cycles[0].atoms[0]
        )
    stitched_cycle = stitched_cycle[0].atoms

    # check the index of pairs of atoms that where seg 1,2 overlap 
    # seg 3 for seg 1, 2 orientation 
    fused_pair_index = []
    for pair in fused_atom_pairs:
        fused_pair_index.append(list(map(
            lambda x: stitched_cycle.index(x),
            pair)))
    result = []
    for i in fused_pair_index:
        if abs(i[0] - i[1]) == 1:
            result.append(min(i))
        else:
            result.append(-1)
    #print(fused_pair_index)
    #print(result)
    result = abs(result[0]-result[1])
    if result == len(stitched_cycle)/2:
        return "0"
    else:
        return "60"

def update_segments_orientation(graph):
    """
    Updated orientation of segments in a molecular graph

    Args:
        graph (MolecularGraph or PolyRingGraph): graph for update
    """
    for segment in graph.segments:
        segment.adjacents["120"] = []
        segment.adjacents["60"] = []
        segment.adjacents["0"] = []

    for i,seg in enumerate(graph.segments):
        overlap(seg,graph.segments[i+1:])

    for i,seg in enumerate(graph.segments):
        for seg2 in graph.segments:
            orien = check_orientation(seg,seg2)
            if not orien:
                continue
            else:
                seg.adjacents[orien].append(seg2)
# %%
"""
#TEST SECTION

from poly_rings.rings import PolyRingGraph

#sample = "C12=CC=CC=C1C3=C(C=CC=C3)C4=C2C=CC=C4"
#sample = "C12=CC=C3C(C(C=CC4=C5C=CC=C4)=C5C6=C3C7=C(C=CC=C7)C=C6)=C1C=CC=C2"
#sample = "C12=CC=CC=C1C=C3C(C=CC=C3)=C2"
#sample = "C12=C(C=CS3)C3=C(C=C(C(C=CS4)=C4C5=C6C=CC7=C5C=CC=C7)C6=C8)C8=C1C(C=CC=C9)=C9C=C2"
sample = "C12=CC=C3C(C(C=CC4)=C4C5=C3C6=C(C=CC=C6)C7=C5C=CC8=C7C=C9C(C=CC9)=C8)=C1CC=C2"
graph = PolyRingGraph(sample)

update_segments_orientation(graph)

for seg in graph.segments:
    print("###############")
    for cyc in seg.cycles:
        if len(cyc) == 5:
            print(55555)
    print(seg)
    print(seg.adjacents)
"""
