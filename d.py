import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd

path = './dataset/convection/test/test_40_convection.csv'
test_data = pd.read_csv(f'{path}').drop(['beta', 'rho', 'nu'], axis=1)

ground_truth_x = test_data['x_data'].to_numpy()
ground_truth_t = test_data['t_data'].to_numpy()
ground_truth_u = test_data['u_data'].to_numpy()

# Assuming the data forms a grid, determine the unique values of x and t
unique_x = np.unique(ground_truth_x)
unique_t = np.unique(ground_truth_t)

# Reshape ground_truth_u to a 2D array
ground_truth_u_2d = ground_truth_u.reshape(len(unique_x), len(unique_t))

# Visualize the solution
fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
extent = [np.min(unique_t), np.max(unique_t), np.min(unique_x), np.max(unique_x)]
sol_img1 = ax.imshow(ground_truth_u_2d, extent=extent, origin='lower', aspect='auto', cmap=cm.jet)

# Add colorbar
cb = fig.colorbar(sol_img1, ax=ax, location='bottom', aspect=20)

# Set axis labels and title
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_title('Heat Map of U values')

plt.show()
