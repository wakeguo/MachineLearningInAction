import numpy as np
import matplotlib.pyplot as plt


a = np.arange(10)
for i in a:
    y1 = i ** 2
    y2 = (i + 1) ** 2
    plt.plot([i, i + 1], [y1, y2])
plt.show()