# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:12:12 2021

@author: Q521100
"""


#%%
# Convert the tree to the reverse Polish expression
def generate_tree2exp(tree):

    if tree.get_type() == 0:
        exp = tree.value

    elif tree.get_type() == 1:
        exp = [generate_tree2exp(tree.left), tree.value]

    elif tree.get_type() == 2:
        exp = [generate_tree2exp(tree.left), generate_tree2exp(tree.right), tree.value]
    # END IF

    return exp


# END DEF
