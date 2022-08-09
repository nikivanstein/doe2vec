# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:15:17 2021

@author: Q521100
"""

import random

import numpy as np

#%%
# % Convert the reverse Polish expression to the function

# %           Meaning                     Syntax
# % 1         Real number in 1-10         1.5
# % 2         Decision vector             (x1,...,xd)
# % 3         First decision variable     x1
# % 4         Translated decision vector  (x2,...,xd,0)
# % 5         Rotated decision vector     XR
# % 6         Index vector                (1,...,d)
# % 7         Random number in 1-1.1      rand()
# % 11        Addition                    x+y
# % 12        Subtraction                 x-y
# % 13        Multiplication              x.*y
# % 14        Division                    x./y
# % 21        Negative                    -x
# % 22        Reciprocal                  1./x
# % 23        Multiplying by 10           10.*x
# % 24        Square                      x.^2
# % 25        Square root                 sqrt(abs(x))
# % 26        Absolute value              abs(x)
# % 27        Rounded value               round(x)
# % 28        Sine                        sin(2*pi*x)
# % 29        Cosine                      cos(2*pi*x)
# % 30        Logarithm                   log(abs(x))
# % 31        Exponent                    exp(x)
# % 32        sum of vector           	  sum(x)
# % 33        Mean of vector          	  mean(x)
# % 34        Cumulative sum of vector	  cumsum(x)
# % 35        Product of vector           prod(x)
# % 36        Maximum of vector           max(x)


#%%
# flatten list of lists recursively
def flatten(list_of_lists):

    if len(list_of_lists) == 0:
        return list_of_lists

    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])

    return list_of_lists[:1] + flatten(list_of_lists[1:])


# END DEF


#%%
# Convert the reverse Polish expression to the function
# dim_x = len(array_x)
# dim_y = array_x.shape[1]
def generate_exp2fun(exp, dim_x, dim_y):

    exp_flat = flatten(exp)

    str_main = []

    for item in exp_flat:

        if item < 0:
            str_main.append(f"(abs({item}))")

        else:
            if item == 1:
                # Real number in 1-10
                str_main.append(str(random.random() * 9 + 1))

            elif item == 2:
                # Decision vector
                str_main.append("array_x")

            elif item == 3:
                # First decision variable
                str_main.append("array_x[:,0]")

            elif item == 4:
                # Translated decision vector
                str_main.append(
                    "(np.vstack((array_x[:,1:].ravel(), np.zeros((len(array_x), 1)).ravel())).T)"
                )

            elif item == 5:
                # Rotated decision vector
                mat_rand = str(np.random.rand(dim_y, dim_y).tolist())
                str_main.append(f"(np.dot(array_x, np.array({mat_rand})))")

            elif item == 6:
                # Index vector
                str_main.append("(np.array(range(1, array_x.shape[1]+1)))")

            elif item == 7:
                # Random number in 1-1.1
                mat_rand = str(np.random.rand(dim_x, 1).tolist())
                str_main.append(f"(1+np.array({mat_rand})/10)")

            elif item == 11:
                # Addition
                str_main = str_main[:-2] + [str_main[-2] + "+" + str_main[-1]]

            elif item == 12:
                # Subtraction
                str_main = str_main[:-2] + [str_main[-2] + "-" + str_main[-1]]

            elif item == 13:
                # Multiplication
                str_main = str_main[:-2] + [str_main[-2] + "*" + str_main[-1]]

            elif item == 14:
                # Division
                str_main = str_main[:-2] + [str_main[-2] + "/" + str_main[-1]]

            elif item == 21:
                # Negative
                str_main = str_main[:-1] + ["-(" + str_main[-1] + ")"]

            elif item == 22:
                # Reciprocal
                str_main = str_main[:-1] + ["1/(" + str_main[-1] + ")"]

            elif item == 23:
                # Multiplying by 10
                str_main = str_main[:-1] + ["10*(" + str_main[-1] + ")"]

            elif item == 24:
                # Square
                str_main = str_main[:-1] + ["np.square(" + str_main[-1] + ")"]

            elif item == 25:
                # Square root
                str_main = str_main[:-1] + ["np.sqrt(abs(" + str_main[-1] + "))"]

            elif item == 26:
                # Absolute value
                str_main = str_main[:-1] + ["abs(" + str_main[-1] + ")"]

            elif item == 27:
                # Rounded value
                # str_main = str_main[:-1] + ['np.round(' + str_main[-1] + ', decimals=5)']
                str_main = str_main[:-1] + ["np.round(" + str_main[-1] + ")"]

            elif item == 28:
                # Sine
                str_main = str_main[:-1] + ["np.sin(2*np.pi*" + str_main[-1] + ")"]

            elif item == 29:
                # Cosine
                str_main = str_main[:-1] + ["np.cos(2*np.pi*" + str_main[-1] + ")"]

            elif item == 30:
                # Logarithm
                str_main = str_main[:-1] + ["np.log(abs(" + str_main[-1] + "))"]

            elif item == 31:
                # Exponent
                str_main = str_main[:-1] + ["np.exp(" + str_main[-1] + ")"]

            elif item == 32:
                # Sum of vector
                str_main = str_main[:-1] + ["np.sum(" + str_main[-1] + ", axis=1)"]

            elif item == 33:
                # Mean of vector
                str_main = str_main[:-1] + ["np.mean(" + str_main[-1] + ", axis=1)"]

            elif item == 34:
                # Cumulative sum of vector
                str_main = str_main[:-1] + ["np.cumsum(" + str_main[-1] + ", axis=1)"]

            elif item == 35:
                # Product of vector
                str_main = str_main[:-1] + ["np.prod(" + str_main[-1] + ", axis=1)"]

            elif item == 36:
                # Maximum of vector
                str_main = str_main[:-1] + ["np.amax(" + str_main[-1] + ", axis=1)"]

            else:
                raise ValueError(f"Operator {item} is not defined!")
            # END IF
        # END IF
    # END FOR

    return str_main[0]


# END DEF


#%%
