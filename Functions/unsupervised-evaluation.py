import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Set the figure size
# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True
#
# # Make a list of columns
columns = ['CHIndex', 'DBIndex', 'Silhouetter score']
#
# # Read a CSV file
df = pd.read_csv("unsupervised-results.csv", usecols=columns)
#
# # Plot the lines
# df.plot()
#
#
# plt.show()
#
# import numpy as np
# import matplotlib.pyplot as plt
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
y1 = df['CHIndex']
y2 = df['DBIndex']
y3 = df['Silhouetter score']
#
# fig, ax1 = plt.subplots()
#
# ax2 = ax1.twinx()
# ax1.plot(x, y1, 'g-')
# ax2.plot(x, y2, 'b-')
# ax2.plot(x, y3, 'r-')
#
# ax1.set_xlabel('X data')
# ax1.set_ylabel('CHIndex', color='g')
# ax2.set_ylabel('DBIndex, Silhouetter score', color='b')
#
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# ax1=plt.subplot(111)
# ax1.plot(np.arange(-1,10),np.arange(-1,10))
# ax2=ax1.twinx()
# ax2.plot(np.arange(10,20),np.arange(100,110))

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

offset = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right", axes=par2,
                                        offset=(offset, 0))

par2.axis["right"].toggle(all=True)

host.set_xlim(1, 15)
host.set_ylim(0, 7)

host.set_xlabel("Number of depression symptoms")
host.set_ylabel("DBIndex")
par1.set_ylabel("CHIndex")
par2.set_ylabel("Silhouetter score")

p1, = host.plot(x, y2, label="DBIndex")
p2, = par1.plot(x, y1, label="CHIndex")
p3, = par2.plot(x, y3, label="Silhouetter score")

par1.set_ylim(0, 200)
par2.set_ylim(0, 1)

host.legend()

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())

plt.draw()
plt.savefig("un-eval")

plt.show()
