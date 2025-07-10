import sys
import glob
import pandas as pd
import os
import time


try:
    sys.path.append(
        glob.glob(
            os.path.expanduser('~/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
                sys.version_info.major,
                sys.version_info.minor,
                'win-amd64' if os.name == 'nt' else 'linux-x86_64'
            ))
        )[0]
    )
except IndexError:
    print("CARLA egg file not found. Please check the path.")
    pass

import carla # pylint: disable=import-error, wrong-import-position

# TODO: Will be used for trajectory analysis

# ===== Setup Client =====
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

try:
    world = client.get_world()
    print(f"Connected to world: {world.get_map().name}")
except RuntimeError as e:
    print(f"Connection failed: {e}")

vehicles = world.get_actors().filter('vehicle.*')
print(f"Found {len(vehicles)} vehicles.")

for v in vehicles:
    print(f"Vehicle ID: {v.id}, Type: {v.type_id}")


# ===== Get All Vehicles =====
vehicles = world.get_actors().filter('vehicle.*')
print(world.get_actors())

if not vehicles:
    print("No vehicles found in the world.")
    exit()

# ===== Prepare Logging =====
all_trajectories = []

# ===== Set number of frames to record =====
num_frames = 200  # about 10 seconds at 20 Hz

print("Recording vehicle trajectories...")

try:
    while True:
        world.tick()  # If not in sync mode, you can use time.sleep(0.05)

        snapshot = world.get_snapshot()
        timestamp = snapshot.timestamp.elapsed_seconds
        frame = snapshot.frame

        for vehicle in vehicles:
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()

            all_trajectories.append({
                'vehicle_id': vehicle.id,
                'frame': frame,
                'timestamp': timestamp,
                'x': transform.location.x,
                'y': transform.location.y,
                'z': transform.location.z,
                'yaw': transform.rotation.yaw,
                'vx': velocity.x,
                'vy': velocity.y,
                'vz': velocity.z
            })

except KeyboardInterrupt:
    print("Stopped recording.")

# ===== Save to CSV =====
df = pd.DataFrame(all_trajectories)
df.to_csv("carla_trajectories.csv", index=False)

print("Trajectory saved to carla_trajectories.csv")
