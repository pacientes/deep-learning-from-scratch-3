# %% simple graph
import numpy as np
import matplotlib.pyplot as plt

# get data
x = np.arange(0, 6, 0.1)  # from 0 to 6 stepsize 0.1
y = np.sin(x)

# plot graph
plt.plot(x, y)
plt.show()

# %% plot two graph in one plot
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# plot graph
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos graph")
plt.legend()
plt.show()

# %% show image
from matplotlib.image import imread

img = imread("cactus.png")
plt.imshow(img)
plt.show()
