#%%
import os, sys

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from data_structures.essential_data import *
from data_structures.tree import Tree, DFS
from data_structures.node import AtomNode
from data_structures.cycles import CycleFinding, Cycle
from data_structures.last_resort import sort_cycle,BruteSearchRing

class MolecularGraph:
    """
    Class for molecular graph.
    Note: the cycles are only generated when 
        the method find_cycles is called.
    """
    def __init__(self,node_class = AtomNode):
        self.tree = []
        self.cycles = []
        self.node_class = node_class

    def from_smiles(self,smiles):
        """
        Constructing the molecular graph from SMILES string.
        Args:
        + smiles (string)
        """
        tokenized_smiles = tokenize(smiles)
        for smiles in tokenized_smiles:
            tree = Tree(node_class=self.node_class)
            tree.from_smiles(smiles)
            fpwsn = FindPairsWithSameNumber()
            tree.traverse(f = fpwsn.f)
            fpwsn.forge_edge()
            self.tree.append(tree)

    def traverse(self,f):
        """
        Traversing the graph and apply function f
        to each of the node.
        Args:
        + f (Python function)
        """
        for tree in self.tree:
            tree.traverse(f)

    def find_cycles(self,mode = None, return_cycles = False):
        """
        Find all cycles (ring) in the graph.
        Each cycle/ring is stored as a Cylce object (using attributes 
        atoms on Cycle to retrieve the list of atoms in cycle). All cycles 
        is stored in attributes cycles.
        Arg:
            + mode:
            + return_cycles (bool): return the list of cycles
        """
        cycles = []
        for tree in self.tree:
            cycle = find_cycles(tree.root)
            cycles += cycle

        if return_cycles:
            return cycles

        self.cycles = cycles

    def list_all_atoms(self):
        result = []
        for tree in self.tree:
            tree.traverse(f=result.append)
        return result

def tokenize(smiles):
    """
    separate the smiles into individual tokens.
    each atomic symbol such as C,H,Br,Cl is a token.
    other symbol such as (,),[,],... is a token.
    """
    sub_smiles_list = []
    tokens_list = []
    for i,char in enumerate(smiles):
        if char in NUMBER:
            tokens_list.append(char)
        elif char in ["(",")","[","]","{","}"]:
            tokens_list.append(char)
        elif char in BOND_CHARACTER.keys():
            tokens_list.append(char)
        elif char in ["+","-"]:
            tokens_list.append(char)
        elif char == "@":
            pass
        elif char in ["/","\\"]:
            pass
        elif char == ".":
            sub_smiles_list.append(tokens_list)
            tokens_list = []
        elif char in ALPHABET:
            if char in UPPER_ALPHABET or char in ["c","o","s","n"]:
                tokens_list.append(char)
            elif char in LOWER_ALPHABET:
                tokens_list[-1] += char
    sub_smiles_list.append(tokens_list)
    return sub_smiles_list

def find_cycles(root,mode = None):
    cf = CycleFinding()
    cycles = cf.find_cycle(root)
    single,compound = sort_cycle(cycles)
    if compound and mode != "minimal":
        bs = BruteSearchRing(single)
        for ring in compound:
            bs.brute_search(ring)
        cycles = bs.single_ring
    return cycles

class FindPairsWithSameNumber:
    """
    Find atoms pair that has the same number in the SMILES,
    then, register connectivity between them.
    """
    def __init__(self):
        self.number2AtomNode = {}
        self.edge = []

    def match(self,num,node):
        """
        Whenever a node which has number beside it in the 
        SMILES string is found, this function is called
        
        If that number has been registered as key in the number2AtomNode 
        dict, the current node and the node in dict value correspond to that key 
        number are registered as a list of two in edge list.
        Also, the key of dictionary is deleted.
        
        If the number has not been registered in the dict, a key of that number is 
        created and point to the current node.
        """
        try:
            match_node = self.number2AtomNode[num]
            self.edge.append([match_node,node])
            del self.number2AtomNode[num]
        except KeyError:
            self.number2AtomNode[num] = node

    def f(self,node):
        """
        Function for apply (the match method of) this class using the tree traversal method
        """
        if node.mark:
            for i in node.mark:
                self.match(i,node)
    
    def forge_edge(self):
        """
        Create connection on the graph from the edge dict.
        """
        for edge in self.edge:
            edge[0].add_connection(edge[1])


#%%
"""
sample = "c1cc2c(cc1)c1c(ccc3ccc(cc13)N(=O)=O)cc2"
graph = MolecularGraph()
graph.from_smiles(sample)

graph.find_cycles()

def f(node):
    print(node,"".join(node.mark),len(node.connectivity.keys()))
    print(node.connectivity)
    print("In cycle?",node.in_cycles)

graph.traverse(f)
"""
# %%
