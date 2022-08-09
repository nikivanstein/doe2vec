# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:39:53 2021

@author: Q521100
"""


#%%
class Node(object):
    def __init__(self, *args):
        self.value = args[0]
        self.left = []
        self.right = []

        if len(args) > 1:
            self.left = args[1]
            if len(args) > 2:
                self.right = args[2]
            # END IF
        # END IF

    #%%
    def get_type(self):
        # Type of node: 0. operand 1. unary operator 2. binary operator
        if not (self.left):
            return 0
        elif not (self.right):
            return 1
        else:
            return 2
        # END IF

    # END DEF

    #%%
    def get_iscons(self):
        # Whether the node is a constant
        if self.value <= 1:
            return True
        else:
            return False
        # END IF

    # END DEF

    #%%
    def get_isscalar(self):
        # Whether the node is a scalar
        if (self.value in [1, 3, 7]) or (self.value <= 1):
            return True
        else:
            return False
        # END IF

    # END DEF

    #%%
    def get_isbinary(self):
        # Whether the node is a binary operator
        if self.value in [11, 12, 13, 14]:
            return True
        else:
            return False
        # END IF

    # END DEF

    #%%
    def get_isvector(self):
        # Whether the node is a vector-oriented operator
        if self.value in [32, 33, 34, 35, 36]:
            return True
        else:
            return False
        # END IF

    # END DEF


#%%
