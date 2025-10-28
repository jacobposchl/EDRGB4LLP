import tonic
from tonic import transforms

dataset = tonic.datasets.DSEC(save_to="./data", split="train", data_selection="events_left")

print(f"Number of samples in the dataset: {len(dataset)}")
events, _ = dataset
print(f"Shape of the events tensor: {events.shape}")

