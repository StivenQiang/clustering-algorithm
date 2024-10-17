import numpy as np
import matplotlib.pyplot as plt

# Function to generate data points around random cluster centers
def generate_cluster_data(n_clusters=7, n_samples=50, cluster_std=0.4):
    centers = np.random.uniform(low=-6, high=6, size=(n_clusters, 2))
    data = []
    for center in centers:
        cluster_data = np.random.normal(loc=center, scale=cluster_std, size=(n_samples, 2))
        data.append(cluster_data)
    return np.vstack(data)

# Generate data
generated_data = generate_cluster_data()

# Save to txt file
output_file = 'testSet3.txt'
np.savetxt(output_file, generated_data, fmt='%.6f', delimiter='\t')

# Plot the generated data
plt.figure(figsize=(10, 6))
plt.scatter(generated_data[:, 0], generated_data[:, 1], s=30, color='blue', alpha=0.6)
plt.title('Generated Cluster Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.axhline(0, color='grey', lw=0.5, ls='--')
plt.axvline(0, color='grey', lw=0.5, ls='--')
plt.show()

# Print the output file name
output_file
