#My first attempt at generating perlin noise. This will be a POS script first.

import numpy as np
import matplotlib.pyplot as plt

#Step 1 make the 2D grid
N = 10
x = np.arange(N+1)
X,Y = np.meshgrid(x,x)

#Step 2 make random gradients sampled from [-1,1]
gradients = np.random.random(size=(N+1, N+1, 2)) * 2 - 1

#Step 3 define a function that returns the perlin noise
def perlin_noise(x, y):
    #Indices of the square the point lies in
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0+1

    #interpolation weights for the linear case
    deltax = x - x0
    deltay = y - y0

    zall = np.zeros((2,2))
    for i, xi in enumerate([x0, x1]):
        for j, yi in enumerate([y0, y1]):
            delta = np.array([x-xi, y-yi])
            grads = gradients[xi, yi]
            zall[i, j] = np.dot(delta, grads)
            continue
        continue
    zyall = zall[0, :] + deltax*(zall[1, :] - zall[0, :])
    return zyall[0] + deltay*(zyall[1] - zyall[0])
    
#Step 4 define a random set of points between 0 and 10 in both directions
xy_points = np.random.random(size = (1000, 2)) * 10
print xy_points
z = np.array([perlin_noise(xi, yi) for xi, yi in xy_points])

#Make a stick plot
#fig, ax = plt.subplots()
#q = ax.quiver(X, Y, gradients[:,:,0], gradients[:,:,1])
#plt.show()

#Make a height plot
fig, ax = plt.subplots()
x, y = xy_points.T
from matplotlib.mlab import griddata
xi = np.linspace(-.1, N+.1, 100)
yi = np.linspace(-.1, N+.1, 100)
zi = griddata(x, y, z, xi, yi, interp='linear')
print zi.shape
CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
CS = plt.contourf(xi, yi, zi, 15,
                  vmax=abs(zi).max(), vmin=-abs(zi).max())
plt.show()
