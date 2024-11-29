import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Define the x range
x = np.linspace(0, 5, 100)

# Define the linear function f(x) = 2x + 1
f_x = 2 * x + 1

# Define the nonlinear function g(x) = x^2
g_x = x**2

# Compute the difference h(x) = g(x) - f(x)
h_x = g_x - f_x

# Define conditions where g(x) > f(x) and f(x) > g(x)
mask_g_greater_f = g_x > f_x
mask_f_greater_g = f_x >= g_x

# Calculate the integrals for each region
integral_g_greater_f = np.trapz(h_x[mask_g_greater_f], x[mask_g_greater_f])
integral_f_greater_g = np.trapz(h_x[mask_f_greater_g], x[mask_f_greater_g])

# Plot the functions
plt.plot(x, f_x, label="f(x) = 2x + 1 (Linear)", color="blue")
plt.plot(x, g_x, label="g(x) = x^2 (Nonlinear)", color="red")

# Fill the areas where g(x) > f(x) and f(x) > g(x)
plt.fill_between(x, f_x, g_x, where=mask_g_greater_f, color='green', alpha=0.3, label=f'Area (g(x) > f(x)): {integral_g_greater_f:.2f}')
plt.fill_between(x, f_x, g_x, where=mask_f_greater_g, color='orange', alpha=0.3, label=f'Area (f(x) > g(x)): {integral_f_greater_g:.2f}')

# Add labels, legend, and title
plt.title("Integral Visualization: Area Between f(x) and g(x)")
plt.xlabel("x")
plt.ylabel("y")

# Annotate the integral values on the plot
plt.text(1.5, 12, f"Integral: {integral_g_greater_f:.2f}", fontsize=12, color="green")
plt.text(3, 5, f"Integral: {integral_f_greater_g:.2f}", fontsize=12, color="orange")

# Show the legend and plot
plt.legend()
plt.show()
