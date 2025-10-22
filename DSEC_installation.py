import tonic
from tonic import transforms

dataset = tonic.datasets.DSEC(save_to="./data", split="train", data_selection="events_left")
sample = dataset[0]

events, image, _ = sample

transform = transforms.ToVoxelGrid(sensor_size=dataset.sensor_size, n_time_bins=4)

voxel = transform(events)

print(voxel.shape, image.shape)