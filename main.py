from nomic import atlas
import numpy as np

# Randomly generate a set of 10,000 high-dimensional embeddings
num_embeddings = 10000
embeddings = np.random.rand(num_embeddings, 256)

# Create Atlas project
dataset = atlas.map_data(embeddings=embeddings)

print(dataset)

# Access your Atlas map and download your embeddings
map = dataset.maps[0]

projected_embeddings = map.embeddings.projected
latent_embeddings = map.embeddings.latent


print(projected_embeddings)

# Access your Atlas map
map = dataset.maps[0]

# Access a pandas DataFrame associating each datum on your map to their topics at each topic depth.
topic_df = map.topics.df

print(map.topics.df)


# Load map and perform vector search for the five nearest neighbors of datum with id "my_query_point"
map = dataset.maps[0]

with dataset.wait_for_dataset_lock():
  neighbors, _ = map.embeddings.vector_search(ids=['my_query_point'], k=5)

# Return similar data points
similar_datapoints = dataset.get_data(ids=neighbors[0])

print(similar_datapoints)