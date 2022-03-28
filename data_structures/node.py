import numpy as np
import os, sys

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from data_structures.essential_data import LOWER_ALPHABET, PERIODIC_TABLE, UPPER_ALPHABET

class Node:
    def __init__(self,value):
        """
        Class for general node in linked list
        Can be used as dummy nodes.
        Attribute:
        + next: point to next node in the linked list
        """
        self.value = value
        self.next = None
        self.random_value = str(np.random.randint(1000000,10000000))
        self.random_repr = str(value) +","+ self.random_value

    def __call__(self):
        return self.value

    def __repr__(self):
        return "BaseNode("+self.random_repr+")"

class AtomNode(Node):
    """
    class for node that represent atom. 
    The value of the node is the symbol of atom represented by node
    Each node contain information on:
    + atomic number (retrieve from PERIODIC_TABLE)
    + atomic mass (same above)
    + connectivity: dictionary of atom: bond for each neighbor 
        atom directly connect to current atom
    + next: (inherited)
    """
    def __init__(
        self, value
        ):
        Node.__init__(self,value)
        if self.value in LOWER_ALPHABET:
            self.AROMATIC = True
        else:
            self.AROMATIC = False
        atom_info = PERIODIC_TABLE[self.value.title()]
        self.atomic_number = atom_info[0]
        self.atomic_mass = atom_info[1]
        self.connectivity = {}
        self.explicit_H = 0
        self.charge = 0
        self.in_cycles = False
        self.mark = []

    def __repr__(self):
        return "AtomNode("+self.random_repr+")"

    def add_connection(
        self,another_atom, number_of_bond=0
        ):
        """
        Add to connectivity of the node that call this function
        the node : bond order. 
        Likewise, add to the connectivity of the node in the 
        arguments the urrent node : bond order

        Args:
        + another_atom (AtomNode): The other atom node
        + number_of_bond (int, optional): bond order. 
            10 for aromtic bond. Defaults to 0.
        """
        if self.AROMATIC == True and another_atom.AROMATIC == True and number_of_bond == 0:
            number_of_bond = 10 #10 for aromatic
        if number_of_bond == 0:
            number_of_bond = 1
        self.connectivity[another_atom] = number_of_bond
        another_atom.connectivity[self] = number_of_bond
