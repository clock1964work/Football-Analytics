import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from mplsoccer import Pitch

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

# Load data
path = r"C:\Users\User\PycharmProjects\soccermatics\passesQ_data.csv"
passes = pd.read_csv(path)



# Load dataset
passes = pd.read_csv(path)

# Function to check if a pass is progressive
def is_progressive(x, y, end_x, end_y):
    start_dist = np.sqrt((100 - x)**2 + (50 - y)**2)  # Distance from goal at start
    end_dist = np.sqrt((100 - end_x)**2 + (50 - end_y)**2)  # Distance from goal at end

    thres = 100  # Default threshold
    if x < 50 and end_x < 50:
        thres = 30  # Own half: must advance at least 30m
    elif x < 50 and end_x >= 50:
        thres = 15  # Moving across halves: must advance at least 15m
    elif x >= 50 and end_x >= 50:
        thres = 10  # Opponent’s half: must advance at least 10m

    return thres <= start_dist - end_dist  # Pass is progressive if it meets the threshold

# Apply function to dataset
passes["is_progressive"] = passes.apply(lambda row: is_progressive(row['x'], row['y'], row['end_x'], row['end_y']), axis=1)

# Keep only Liverpool passes made by Trent Alexander-Arnold
#liverpool_passes = passes.loc[(passes["name"] == "Mohamed Salah") & (passes["match_id"] == 1821142)]

#liverpool_passes = passes.loc[passes["name"] == "Mohamed Salah"]
liverpool_passes = passes.loc[passes["match_id"] == 1821142]
liverpool_passes = liverpool_passes[~liverpool_passes["qualifiers"].astype(str).str.contains("ThrowIn", na=False)]

# Exclude the next row after a corner
corner_indices = liverpool_passes[liverpool_passes["qualifiers"].astype(str).str.contains("CornerTaken", na=False)].index
liverpool_passes = liverpool_passes[~liverpool_passes.index.isin(corner_indices)]

# Keep only progressive passes (ball moves forward + meets distance rule)
liverpool_progressive = liverpool_passes[liverpool_passes["is_progressive"] == True]

# Compute pass angles
liverpool_progressive["angle"] = np.arctan2(
    liverpool_progressive["end_y"] - liverpool_progressive["y"],
    liverpool_progressive["end_x"] - liverpool_progressive["x"]
)

# Prepare data for clustering
X = liverpool_progressive[["x", "y", "end_x", "end_y", "angle"]].values


# Elbow method to find the optimal number of clusters
K = np.linspace(1, 20, 20)
elbow = {"sse": [], "k": [], "sil": []}
for k in K:
    cluster = KMeans(n_clusters=int(k), random_state=2147)
    labels = cluster.fit_predict(X)
    elbow["sse"].append(cluster.inertia_)  # SSE (Sum of Squared Errors)
    elbow["k"].append(k)

# Plotting the Elbow graph (SSE vs K)
plt.scatter(elbow["k"], elbow["sse"])
plt.plot(elbow["k"], elbow["sse"])
plt.xticks(np.linspace(1, 20, 20))
plt.xlabel("K (Number of Clusters)")
plt.ylabel("SSE (Sum of Squared Errors)")
plt.title("Elbow Method for Optimal K")
plt.show()


# Function to compute inertia (average distance within clusters)
def compute_inertia(a, X):
    W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    return np.mean(W)


# GAP statistic function
def compute_gap(clustering, data, k_max, n_references=5):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    reference = np.random.rand(*data.shape)  # Create random reference data for GAP calculation
    reference_inertia = []
    ondata_inertia = []

    for k in range(1, k_max + 1):
        print(f"Processing k = {k}...")  # Track progress
        local_inertia = []

        # Compute inertia for the reference data
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference))

        reference_inertia.append(np.mean(local_inertia))

        # Compute inertia for the actual data
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))

    gap = np.log(reference_inertia) - np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)


# Run GAP statistic calculation with the KMeans clustering
k_max = 10  # Maximum number of clusters to test for GAP statistic
gap, reference_inertia, ondata_inertia = compute_gap(KMeans(random_state=2147), X, k_max)

# Plot GAP statistic result
plt.plot(range(1, k_max + 1), gap, '-o')
plt.ylabel('GAP Statistic')
plt.xlabel('Number of Clusters (k)')
plt.title('Optimal Cluster Selection using GAP Statistic')
plt.show()

# Print computed values
print("GAP Values:", gap)
print("Reference Inertia:", reference_inertia)
print("On-Data Inertia:", ondata_inertia)

# Run K-Means clustering with k = 4
k = 9
cluster = KMeans(n_clusters=k, random_state=2147)
labels = cluster.fit_predict(X)

# Assign cluster labels to Liverpool's progressive passes
liverpool_progressive["label"] = labels


# Set up the pitch
pitch = Pitch(line_color='black', pitch_type="opta")
fig, axs = pitch.grid(ncols=3, nrows=3, grid_height=0.85, title_height=0.06, axis=False,
                      endnote_height=0.04, title_space=0.04, endnote_space=0.01)

# Plot each cluster
for clust, ax in zip(range(k), axs['pitch'].flat[:k]):
    # Adjust cluster title position: a little lower so it doesn't touch the boundary line
    ax.text(0.5, 1.00, "Cluster " + str(int(clust + 1)),  # 1.02 moves the title further down from the top
            ha='center', va='bottom', fontsize=13, transform=ax.transAxes)

    # Get passes belonging to this cluster
    clustered = liverpool_progressive[liverpool_progressive["label"] == clust]

    # Split successful and unsuccessful passes
    successful_passes = clustered[clustered["outcome_type_display_name"] == "Successful"]
    unsuccessful_passes = clustered[clustered["outcome_type_display_name"] == "Unsuccessful"]

    # Plot successful passes (green)
    pitch.scatter(successful_passes.x, successful_passes.y, alpha=0.5, s=50, color="green", ax=ax)
    pitch.arrows(successful_passes.x, successful_passes.y,
                 successful_passes.end_x, successful_passes.end_y, color="green", ax=ax, width=1)

    # Plot unsuccessful passes (red)
    pitch.scatter(unsuccessful_passes.x, unsuccessful_passes.y, alpha=0.5, s=50, color="red", ax=ax)
    pitch.arrows(unsuccessful_passes.x, unsuccessful_passes.y,
                 unsuccessful_passes.end_x, unsuccessful_passes.end_y, color="red", ax=ax, width=1)

# Set the main title for the entire figure
axs['title'].text(0.5, 0.5, 'Liverpool Progressive Passes Clusters Against Newcastle',
                  ha='center', va='center', fontsize=25)

plt.show()



