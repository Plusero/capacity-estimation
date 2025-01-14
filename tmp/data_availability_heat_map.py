import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Creating dummy data
systems = [f'ID{str(i).zfill(3)}' for i in range(1, 88)]  # System IDs
dates = pd.date_range(start='2014-01', end='2017-12',
                      freq='M').strftime('%Y-%m')  # Monthly intervals
# Random availability percentages
data = np.random.randint(0, 101, size=(len(systems), len(dates)))

# Convert to DataFrame
df = pd.DataFrame(data, index=systems, columns=dates)
print(f"df.shape: {df.shape}")
print(f'columns: {df.columns}')
# Step 3: Plot the heatmap
plt.figure(figsize=(15, 10))
ax = sns.heatmap(df, cmap='YlGnBu', cbar_kws={
    'label': 'Data availability rate [%]'},
    linewidths=0.1, linecolor='gray')
# Adjust the colorbar label size
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_size(30)  # Set label size
plt.title('Data Availability Heatmap', fontsize=30)
plt.xlabel('Time', fontsize=12)
plt.ylabel('System', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(fname='data_availability_heat_map.png')
plt.close()
