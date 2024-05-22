from matplotlib import pyplot as plt
import numpy as np

x = [3, 5, 10,20,30]
y = [301.53, 214.14, 135.69, 62.83, 39.15]
ci = 1.96 * np.std(y)/np.sqrt(len(x))
print(ci)

fig, ax = plt.subplots()
ax.set_ylabel('Iterations/s')
ax.set_xlabel('Number of Endmembers in X')
ax.plot(x, y)
ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)
plt.show()
plt.clf()
plt.close()

x = [10, 25, 50, 100, 500, 1000]
y = [3.25, 3.72, 2.97, 2.72, 1.25, 0.77]
ci = 1.96 * np.std(y)/np.sqrt(len(x))
print(ci)

fig, ax = plt.subplots()
ax.set_ylabel('Iterations/s')
ax.set_xlabel('Number of MESMA models')
ax.plot(x, y)
ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)
plt.show()
plt.clf()
plt.close()