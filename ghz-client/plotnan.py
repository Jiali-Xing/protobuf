import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example data
x = np.arange(10)
y = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10])

# Create DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Interpolate to fill NaN values
df['y'] = df['y'].interpolate()

# Plotting
plt.plot(df['x'], df['y'], marker='o')
plt.title('Continuous Line Plot with Interpolated NaN values')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
