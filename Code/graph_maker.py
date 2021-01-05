import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


plt.style.use('seaborn-darkgrid')

cov_dist = [38,28,71,6,26,47,67,10,43,31]
norm_dist = [546,399,95,177,66,44,107,50,102,16]
pne_dist = [416,573,834,817,908,909,826,940,855,953]

epoch = np.arange(len(cov_dist))

fig,ax = plt.subplots(1,1)
ax.set_title("Sample Distribution Per Epoch")

ax.plot(epoch,cov_dist, label="Covid_19")
ax.plot(epoch, norm_dist, label="Normal")
ax.plot(epoch, pne_dist, label="Pneumonia")
ax.legend()
plt.show()