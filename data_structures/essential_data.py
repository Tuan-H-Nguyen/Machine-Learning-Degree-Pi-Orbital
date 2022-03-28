import string

PERIODIC_TABLE = {
    "H":[1,1],
    "C":[6,12], "N":[7,14], 
    "S":[16,32], "O":[8,16],
    "F":[9,19], "Cl":[17,35.45], 
    "Br":[35,80], "I":[53,127]
}
BOND_CHARACTER = {
    "-":1,"=":2, "#":3
}

VALENCE = {
    "H":[1],"C":[4], "N":[3,5], 
    "S":[6], "O":[2],
    "F":[1], "Cl":[1,3,5,7], 
    "Br":[1,3,5,7],"I":[1,3,5,7]
}

def get_valence(atom_node):
    """
    Get the value of valence that
    is the smallest value and larger 
    than the connectivity of the node
    """
    valence = sorted(VALENCE[atom_node.value])
    for i,value in enumerate(valence):
        if i == len(valence) - 1:
            return value
        elif value > len(atom_node.connectivity.keys()):
            break
    return valence[i+1]

UPPER_ALPHABET = string.ascii_uppercase
LOWER_ALPHABET = string.ascii_lowercase 
ALPHABET = UPPER_ALPHABET + LOWER_ALPHABET
NUMBER = [str(i) for i in range(20)]
