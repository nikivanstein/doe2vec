# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:53:17 2021

@author: Q521100
"""


import random

import numpy as np

from .NODE import Node


#%%
def all_ismember(list_item, list_check):

    boolean = True

    for item in list_item:
        if item not in list_check:
            boolean = False
            break
        # END IF
    # END FOR

    return boolean


# END DEF


#%%
# Randomly generate a tree
def generate_tree(minlen, maxlen):

    # The initial tree
    tree = Node(33, Node(2))

    # Probabilities of selecting each operand
    pOperand = np.array([30, 5, 1, 1, 1, 1, 0])
    pOperand = np.cumsum(pOperand)
    pOperand = pOperand / np.max(pOperand)

    # Indexex of each operand
    mOperand = np.array(range(1, 8))

    # Probabilities of selecting each operator
    pOperator = [15, 15, 10, 10, 2, 2, 2, 5, 5, 2, 2, 3, 3, 3, 3, 3, 3, 1, 1, 1]
    pOperator = np.cumsum(pOperator)
    pOperator = pOperator / np.max(pOperator)

    # Indexes of each operator
    mOperator = np.concatenate((np.arange(11, 15), np.arange(21, 37)), axis=None)

    # Randomly generate a tree
    for i in range(random.randint(minlen, maxlen)):
        # Randomly reach a leaf
        p = tree
        while p.get_type() > 0:
            if (p.get_type() == 1) or (random.random() < 0.5):
                p = p.left
            else:
                p = p.right
            # END IF
        # END WHILE

        # Select an operator
        operator = int(mOperator[np.argwhere(random.random() <= pOperator)[0][0]])
        # Binary operator
        if operator <= 20:
            # Select an operand
            operand = int(mOperand[np.argwhere(random.random() <= pOperand)[0][0]])
            if random.random() < 0.5:
                p.left = Node(p.value)
                p.right = Node(operand)
            else:
                p.left = Node(operand)
                p.right = Node(p.value)
            # END IF
            p.value = operator
        # Unary operator
        else:
            p.left = Node(p.value)
            p.value = operator
        # END IF
    # END FOR

    tree = injection(tree)
    cleaning1(tree)
    cleaning2(tree)
    cleaning1(tree)
    cleaning2(tree)

    return tree


# END DEF


#%%
# Difficulty injection
def injection(tree):

    r = random.random()
    if r < 0.05:
        # Noisy landscape
        tree = Node(13, tree, Node(7))
    elif r < 0.1:
        # Flat landscape
        tree = Node(27, tree)
    elif r < 0.2:
        # Multimodal landscape
        tree = Node(11, tree, Node(28, tree))
    elif r < 0.25:
        # Highly multimodal landscape
        tree = Node(11, tree, Node(23, Node(28, tree)))
    elif r < 0.3:
        # Linkages between all the variables and the first variable
        tree = injection2(tree, 1)
    elif r < 0.35:
        # Linkages between each two contiguous variables
        tree = injection2(tree, 2)
    elif r < 0.4:
        # Complex linkages between all the variables
        tree = injection2(tree, 3)
    elif r < 0.45:
        # Different optimal values to all the variables
        tree = injection2(tree, 4)
    # END IF

    return tree


# END DEF


#%%
# Difficulty injection 2
def injection2(tree, tree_type):

    if isinstance(tree, Node):
        if tree.value == 2:
            if tree_type == 1:
                tree = Node(12, Node(2), Node(3))
            elif tree_type == 2:
                tree = Node(12, Node(2), Node(4))
            elif tree_type == 3:
                tree = Node(5)
            elif tree_type == 4:
                tree = Node(13, Node(6), Node(2))
            # END IF
        else:
            tree.left = injection2(tree.left, tree_type)
            tree.right = injection2(tree.right, tree_type)
        # END IF
    # END IF

    return tree


# END DEF


#%%
# Clean the unary operators
def cleaning1(tree):

    if tree.get_type() == 0:
        scalar = tree.get_isscalar()

    elif tree.get_type() == 1:
        scalar = cleaning1(tree.left)

        if scalar and tree.get_isvector():
            # If the node is a vector-oriented operator and the child is a scalar, replace the node with its child
            tree.value = tree.left.value
            tree.right = tree.left.right
            tree.left = tree.left.left
        else:
            if tree.value == 26 and (tree.left.value in [25, 26, 30]):
                # If the node is abs and the child is abs, log, or sqrt, replace the node with its child
                tree.value = tree.left.value
                tree.left = tree.left.left

            elif (tree.value in [25, 26, 30]) and tree.left.value == 26:
                # If the node is abs, log, or sqrt and the child is abs, replace the child with its child
                tree.left = tree.left.left

            elif (
                tree.value == 21
                and tree.left.value == 21
                or tree.value == 22
                and tree.left.value == 22
            ):
                # If both the node and child are negative or reciprocal, replace the node with its child's child
                tree.value = tree.left.left.value
                tree.right = tree.left.left.right
                tree.left = tree.left.left.left

            elif (
                tree.value == 24
                and tree.left.value == 25
                or tree.value == 25
                and tree.left.value == 24
                or tree.value == 30
                and tree.left.value == 31
                or tree.value == 31
                and tree.left.value == 30
            ):
                # If both the node and child are square and sqrt or log and exp, replace the node with its child's child
                tree.value = tree.left.left.value
                tree.right = tree.left.left.right
                tree.left = tree.left.left.left

            elif tree.value == 21 and tree.left.value == 12:
                # If the node is negative and the child is subtraction, replace the node with its child then exchange its children
                tree.value = 12
                tree.right = tree.left.left
                tree.left = tree.left.right

            elif tree.value == 22 and tree.left.value == 14:
                # If the node is reciprocal and the child is division, replace the node with its child then exchange its children
                tree.value = 14
                tree.right = tree.left.left
                tree.left = tree.left.right
            # END IF
        # END IF
        scalar = scalar or tree.get_isvector() and tree.value != 34

    elif tree.get_type() == 2:
        scalar1 = cleaning1(tree.left)
        scalar2 = cleaning1(tree.right)
        scalar = scalar1 and scalar2
    # END IF

    return scalar


# END DEF


#%%
# Clean the binary operators
def cleaning2(tree):

    if tree.get_type() == 0:
        cons = tree.get_iscons()

    elif tree.get_type() == 1:
        cons = cleaning2(tree.left)

    elif tree.get_type() == 2:
        cons1 = cleaning2(tree.left)
        cons2 = cleaning2(tree.right)
        cons = cons1 and cons2

        if cons:
            # If both the children are constants, change the node to a constant
            tree.value = 1
            tree.left = []
            tree.right = []

        elif tree.right.value == 21:
            # If the node is addition and the right child is negative, change the node to subtraction and replace the right child with its child
            if tree.value == 11:
                tree.value = 12
                tree.right = tree.right.left
            # If the node is subtraction and the right child is negative, change the node to addition and replace the right child with its child
            elif tree.value == 12:
                tree.value = 11
                tree.right = tree.right.left
            # END IF

        elif tree.right.value == 22:
            # If the node is multiplication and the right child is reciprocal, change the node to division and replace the right child with its child
            if tree.value == 13:
                tree.value = 14
                tree.right = tree.right.left
            # If the node is division and the right child is reciprocal, change the node to multiplication and replace the right child with its child
            elif tree.value == 14:
                tree.value = 13
                tree.right = tree.right.left
            # END IF

        elif tree.left.value == 21 and tree.value == 11:
            # If the node is addition and the left child is negative, change the node to subtraction, replace the left child
            # with its child, then exchange the node's children
            tree.value = 12
            temp = tree.right
            tree.right = tree.left.left
            tree.left = temp

        elif tree.left.value == 22 and tree.value == 13:
            # If the node is multiplication and the left child is reciprocal, change the node to division, replace the left
            # child with its child, then exchange the node's children
            tree.value = 14
            temp = tree.right
            tree.right = tree.left.left
            tree.left = temp

        elif tree.left.get_isbinary() and cons2:
            if all_ismember([tree.value, tree.left.value], [11, 12]) or all_ismember(
                [tree.value, tree.left.value], [13, 14]
            ):
                if tree.left.left.get_iscons() and tree.left.right.get_iscons():
                    # If the left child is a binary operator, at least one of the left child's children is a constant,
                    # and the right child is a constant, replace the node with its left child
                    tree.value = tree.left.value
                    tree.right = tree.left.right
                    tree.left = tree.left.left
                # END IF
            # END IF

        elif cons1 and tree.right.get_isbinary():
            if all_ismember([tree.value, tree.right.value], [11, 12]) or all_ismember(
                [tree.value, tree.right.value], [13, 14]
            ):
                if tree.right.right.get_iscons():
                    # If the right child is a binary operator, the right child's right child is a constant, and the
                    # left child is a constant, replace the right child with its left child
                    tree.right = tree.right.left
                elif tree.right.left.get_iscons():
                    # If the right child is a binary operator, the right child's left child is a constant, and the
                    # left child is a constant, replace the right child with its right child then change the node's operator
                    if tree.value == tree.right.value:
                        if tree.value <= 12:
                            tree.value = 11
                        else:
                            tree.value = 13
                        # END IF
                    else:
                        if tree.value <= 12:
                            tree.value = 12
                        else:
                            tree.value = 14
                        # END IF
                    # END IF
                    tree.right = tree.right.right
                # END IF
            # END IF

        elif tree.left.get_isbinary() and tree.right.get_isbinary():
            if all_ismember(
                [tree.left.value, tree.value, tree.right.value], [11, 12]
            ) or all_ismember(
                [tree.left.value, tree.value, tree.right.value], [13, 14]
            ):
                if (
                    tree.left.left.get_iscons() or tree.left.right.get_iscons()
                ) and tree.right.right.get_iscons():
                    # If both the left and right children are binary operators, at least one of the left child's children is a constant,
                    # and the right child's right child is a constant, replace the right child with its left child
                    tree.right = tree.right.left
                elif (
                    tree.left.left.get_iscons() or tree.left.right.get_iscons()
                ) and tree.right.left.get_iscons():
                    # If both the left and right children are binary operators, at least one of the left child's children is a constant,
                    # and the right child's left child is a constant, replace the right child with its right child then change the node's operator
                    if tree.value == tree.right.value:
                        if tree.value <= 12:
                            tree.value = 11
                        else:
                            tree.value = 13
                        # END IF
                    else:
                        if tree.value <= 12:
                            tree.value = 12
                        else:
                            tree.value = 14
                        # END IF
                    # END IF
                    tree.right = tree.right.right
                # END IF
            # END IF
        # END IF
    # END IF

    return cons


# END DEF


#%%
