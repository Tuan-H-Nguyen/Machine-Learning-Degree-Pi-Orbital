#%%
import os, sys,copy

from sympy import Poly

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from poly_rings.rings import PolyRingGraph

from utils.logic import intersection

def check_thiophene(segment):
    """
    Check if a thiophene ring is in a segment.
    Args:
    + segment (Segment):
    Returns:
    + bool: True if thiophene ring in the segment
        False otherwise
    """
    for ring in segment.cycles:
        if len(ring) == 5:
            return True

def reference_segment(segment,thiophene=False):
    """
    Get the contribution to the DPO value of the 
    reference segment.

    Args:
    + segment (Segment): reference segment
    + thiophene (bool, optional): If thiophene in the
        the segment. Defaults to False.

    Returns:
        String: polynomials contribution of the 
            reference segment.
    """
    n = len(segment) - 1
    a = 0
    for i in range(n):
        a += i
    if thiophene:
        return ["{} - {}*a - af".format(n,a)]
    else: return ["{} - {}*a".format(n,a)]

def b_segment(segment):
    """
    Get the contribution to the DPO value of the 
    segment that forms an angle of 120 with the 
    reference segment.

    Args:
        segment (Segment): segment whose DPO polynomial
            contribution is formulated.

    Returns:
        String: polynomials contribution of the 
            reference segment.
    """

    n = len(segment) - 1
    thiophene = check_thiophene(segment)
    if thiophene and n == 1:
        return []
    overlayer = -1
    result = []
    d = "df" if thiophene else "d"
    for i in range(n):
        overlayer += 1
        result.append("b*{}**{}".format(d,overlayer))
    return result

def c_segment(
    segment,main_segment
    ):
    """
    Get the contribution to the DPO value of the 
    segment that forms an angle of 120 with the 
    reference segment.

    Args:
        segment (Segment): segment whose DPO polynomial
            contribution is formulated.
        main_segment (Segment): reference segment that
            segment forms an angle of 60 with.

    Returns:
        String: polynomials contribution of the 
            reference segment.
    """

    b_seg = intersection(
        segment.adjacents["120"],
        main_segment.adjacents["120"]
        )
    
    assert len(b_seg) == 1
    b_seg = b_seg[0]
    overlayer = len(b_seg) - 1

    n = len(segment) - 1
    
    thiophene = check_thiophene(segment)
    
    if thiophene and n == 1 and overlayer == 1:
        return []

    d = "df" if thiophene else "d"

    result = []
    for i in range(n):
        result.append("c*{}**{}".format(d,overlayer-1))
        overlayer += 1
    return result

def d_segment(
    segment,main_segment
    ):
    """
    Get the contribution to the DPO value of the 
    segment that forms an angle of 120 with the 
    reference segment.

    Args:
    + segment (Segment): segment whose DPO polynomial
        contribution is formulated.
    + main_segment (Segment): reference segment that
        segment forms an angle of 60 with.

    Returns:
    + String: polynomials contribution of the 
        reference segment.
    """

    b_seg = intersection(
        segment.adjacents["120"],
        main_segment.adjacents["120"]
        )
    assert len(b_seg) == 1
    b_seg = b_seg[0]
    overlayer = len(b_seg) - 1
    thiophene = check_thiophene(segment)
    d = "df" if thiophene else "d"

    result = []
    result.append(
        "(" + reference_segment(segment)[0] + ")" 
        + "*{}**{}".format(d,overlayer))    
    return result

def DPO_generate(smiles):
    """
    Basic function for convert smiles -> dpo polynomials
    Args:
    + smiles (string): smiles string of either PAH or 
        thienoacenes
    Returns:
    + String: Truncated DPO polynomial.
    """
    DPO_list = []
    graph = PolyRingGraph(smiles)
    segments = sorted(
        graph.segments,key=len,reverse=True)
    highest_len = len(segments[0])
    for seg in segments:
        if len(seg) == highest_len:
            dpo = []
            dpo += reference_segment(
                seg,thiophene=check_thiophene(seg))
            for seg2 in seg.adjacents["120"]:
                dpo += b_segment(seg2)
            for seg2 in seg.adjacents["60"]:
                dpo += c_segment(seg2,seg)
            for seg2 in seg.adjacents["0"]:
                dpo += d_segment(seg2,seg)
            DPO_list.append(" + ".join(dpo))
    DPO_list = ["(" + dpo + ")" for dpo in DPO_list]
    divider = "*(1/{})".format(len(DPO_list))
    DPO_list = "(" + " + ".join(DPO_list) + ")"
    return DPO_list + divider


