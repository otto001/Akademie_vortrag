# Load the libraries you will need
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import perceptron
from pandas import *

inputs = DataFrame({
'x' : [2, 1, 2, 5, 7, 2, 3, 6, 1, 2, 5, 4, 6, 5],
'y' : [2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7],
'Targets' : [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]
})


user = DataFrame({
'x' : [0.0],
'y' : [0.0],
'R' : [0]
}, index=['R', 'x', 'y'])

# Set an array of colours, we could call it
# anything but here we call is colormap
# It sounds more awesome
colormap = np.array(['r', 'g'])


net = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)

# Train the perceptron object (net)
net.fit(inputs[['x', 'y']], inputs['Targets'])

# Plot the original data


#plt.show()

while True:

    plt.scatter(inputs.x, inputs.y, facecolors=colormap[inputs.Targets], s=60)
    plt.scatter(user.x, user.y, facecolors=colormap[user.R], s=60, edgecolors='k', lw = 2)

    plt.draw()
    plt.pause(0.001)

    x = input()
    y = input()


    result = net.predict([[float(x), float(y)]])

    temp = DataFrame({
        'x': [float(x)],
        'y': [float(y)],
        'R': [int(result[0])]
    }, index=['R', 'x', 'y'])

   # user.loc[user.index.max() + 1] = [int(result[0]), float(x), float(y)]
    user = user.append(temp)
    print()
    #plt.show(block=False)

plt.show()