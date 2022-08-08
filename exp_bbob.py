import bbobbenchmarks as bbob
from doe2vec import Doe2Vec
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib.pyplot as plt
from matplotlib import cm

def createSurfacePlot(bbob_fun, gen_fun, name="bbobx"):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    
    # BBOB fun
    X = np.arange(-5, 5, 0.1)
    Y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(X, Y)
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    z1 = np.asarray(list(map(bbob_fun, positions))).reshape(100,100)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(X, Y, z1, cmap=cm.coolwarm,
                        linewidth=1, antialiased=True)
    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    #Generated fun
    X2 = np.arange(0, 1, 0.01)
    Y2 = np.arange(0, 1, 0.01)
    X2, Y2 = np.meshgrid(X2, Y2)
    array_x = np.vstack([X2.ravel(), Y2.ravel()]).T
    z2 = eval(gen_fun)
    z2 = np.array(z2).reshape(100,100)
    #second
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(X2, Y2, z2, cmap=cm.coolwarm,
                        linewidth=1, antialiased=True)

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(f"{name}.png")



obj = Doe2Vec(2, 8, n=1000000, latent_dim=16, use_mlflow=False)
if not obj.load():
    obj.generateData()
    obj.compile()
    obj.fit(100)
    obj.save()
sample = obj.sample * 10 - 5

for f in range(1,25):
    for i in range(5):
        fun, opt = bbob.instantiate(f,i)
        name = f"plots-big-2-8-16/bbob-f-{f}-i-{i}"
        bbob_y =  np.asarray(list(map(fun, sample)))
        array_y = (bbob_y.flatten() - np.min(bbob_y)) / (
                    np.max(bbob_y) - np.min(bbob_y)
                )
        encoded = obj.encode([array_y])
        gen_fun, dist = obj.getNeighbourFunction(encoded)
        print(f, i, dist)
        createSurfacePlot(fun, gen_fun, name)