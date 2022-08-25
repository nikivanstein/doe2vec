# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:28:46 2022

@author: Q521100

Uses https://github.com/fx-long/CEOELA
Clone this repo in this folder before running.
"""



from CEOELA.CEOELA import CEOELA_pipeline


import os
filename = 'DOE_data.xlsx'


def run_ELA(filename='DOE_data.xlsx', label="DOE"):
    filepath = os.path.join(os.getcwd(), filename)
    #%%
    # initliaze
    ceoela_pipeline = CEOELA_pipeline(filepath,
                                    list_sheetname = [],
                                    problem_label = label,
                                    filepath_save = '',
                                    bootstrap = False,
                                    bootstrap_size = 0.8,
                                    bootstrap_repeat = 2,
                                    bootstrap_seed = 0,
                                    BBOB_func = ['F1'],
                                    BBOB_instance = [1],
                                    BBOB_seed = 0,
                                    AF_number = 2,
                                    AF_seed = 0,
                                    np_ela = 8,
                                    purge = True,
                                    verbose = True,
                                    )

    #%%
    # data pre-processing
    ceoela_pipeline.DataPreProcess()

    #%%
    # computation of ELA features
    ceoela_pipeline.ComputeELA(ELA_problem=True, ELA_BBOB=False, ELA_AF=False)



run_ELA('ela-d2.xlsx', 'd2')







#%%












