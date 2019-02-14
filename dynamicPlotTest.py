import pylab
import time
import random
import numpy as np
import matplotlib.pyplot as plt

dat=[0,1]
x,y=[0],[0]
fig = plt.figure()
ax = fig.add_subplot(111)
Ln, = ax.plot(dat)
plt.ion()
plt.show()
for i in range (18):
    x.append(np.sin(i))
    y.append(np.cos(i))
    print(x)
    print(y)
    Ln.set_ydata(y)
    Ln.set_xdata(x)
    plt.pause(1)

    print('done with loop')