#%%
import os, sys

from numpy import true_divide

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from data_structures.node import AtomNode
from data_structures.essential_data import *

class State:
    def __init__(self):
        self.idx = 0
        self.bond_flag = 0

class Tree:
    """
    Class for the tree. The SMILES string can be thought of
    as a tree data structure, acyclic undirected graph. This 
    tree can be constructed recursively. The tree has the first 
    atom in the string as the root, then goes down, branching
    as parentheses appear.

    Note: the tree can be constructed back to the SMILES string by
    Depth First Search algorithm.
    
    This class also has implemention for DFS in the tree. Each node 
    is visit and applied (not implemented) function f. To use, write 
    a function f that take node as input, and call self.traverse(f).
    """
    def __init__(
        self,
        node_class=AtomNode
        ):
        self.node_class = node_class
        self.root = None

    def from_smiles(self,smiles):
        self.smiles = smiles
        state = State()
        self.root, state = self.atom_spec(
            self.smiles[state.idx],state
        )
        state = self.trace_tree(self.root,state=state)
        return state

    def trace_tree(
        self, prev_node,
        state):
        """
        Recursively create the tree from 
        n_1(n_2(n_3)n_4(n_5))n_6... where n_i is the i-th 
        sub-string of the SMILES string. 
        """
        branch_flag = False
        while state.idx < len(self.smiles):
            char = self.smiles[state.idx]
            if char == '(':
                branch_flag = True
                state.idx += 1
            elif char.title() in PERIODIC_TABLE.keys() or char == "[":
                curr_node, state = self.atom_spec(char,state)
                prev_node.add_connection(curr_node,state.bond_flag)
                state.bond_flag = 0
                if branch_flag:
                    state = self.trace_tree(curr_node,state)
                    branch_flag = False
                else:
                    prev_node = curr_node                
            elif char in NUMBER:
                prev_node.mark.append(char)
                state.idx += 1
            elif char in BOND_CHARACTER:
                state.bond_flag = BOND_CHARACTER[char]
                state.idx += 1
            elif char == ")":
                state.idx += 1
                return state
            else:
                raise Exception("Unhandled case!",char)

    def atom_spec(self,char,state):
        """
        Record new atom. This function accept an atom token 
        (e.g. "c","Br",...) or everything between the bracket
        (e.g. "[nH]") and return atom and the state object.
        """
        if char.title() in PERIODIC_TABLE.keys():
            #atom specification without the bracket
            state.idx += 1
            return self.node_class(char),state
        
        h_flag = False
        charge_flag = False

        #atom specification with bracket. Inside the bracket
        #the charge and attached hydrogen atom maybe specified
        while self.smiles[state.idx] != "]":
            state.idx += 1
            char = self.smiles[state.idx]
            if char == "H":
                try:
                    curr_node.explicit_H = 1
                    h_flag = True
                except UnboundLocalError:
                    curr_node = self.node_class(char)
            elif char.title() in PERIODIC_TABLE.keys():
                curr_node = self.node_class(char)
            elif char in ["+","-"]:
                curr_node.charge += int(char+"1")
                charge_flag = True
            elif char in NUMBER:
                if charge_flag:
                    curr_node.charge = curr_node.charge*int(char)
                    charge_flag = False
                elif h_flag:
                    curr_node.explicit_H = curr_node.explicit_H*int(char)
                    h_flag = False
        state.idx += 1 #step through the "]"
        return curr_node,state


    def traverse(self,f):
        dfs = DFS()
        dfs.traverse(self.root,f)

class DFS:
    """
    Depth first search
    """
    def __init__(self):
        self.visited_nodes = []
    def visit(self,node,f):
        """
        Visit node, do something with it,
        mark it, then moving on to nodes that
        connect to it

        Args:
            node (_type_): _description_
            f (_type_): _description_
        """
        if node in self.visited_nodes:
            return
        f(node)
        self.visited_nodes.append(node)
        for i in node.connectivity.keys():
            self.visit(i,f)

    def traverse(self,root,f=print):
        self.visit(root,f)

