import glob
import os
import sys
import math
# find carla module
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
from carla import ColorConverter as cc
import numpy as np
import time

running = True
actor_list = []
point = 0
# When spawning with attachment, location must be relative to the parent actor.
# distence:5m 10m 15m
# angle: 0 45 90 135 180 225 270 315
# pitch: 0 15 30
distences = [5.0, 10.0, 15.0]
angles = [0, math.pi*1/4, math.pi*1/2, math.pi*3/4, math.pi, math.pi*5/4, math.pi*3/2, math.pi*7/4]
pitches = [0, math.pi*1/12, math.pi*1/6]
relative_location = []
relative_rotation = []
relative_transform = []
for dis in distences:
    for angle in angles:
        for pitch in pitches:
            # camera location relative to car
            relative_x = dis * math.cos(pitch) * math.cos(angle)
            relative_y = dis * math.cos(pitch) * math.sin(angle)
            relative_z = dis * math.sin(pitch)
            relative_location.append([relative_x, relative_y, relative_z])
            # camera rotation relative to car
            reletive_pitch = 0 - math.degrees(pitch)
            relative_yaw = math.degrees(angle) - 180
            relative_rotation.append([reletive_pitch, relative_yaw])

            relative_transform.append([[relative_x, relative_y, relative_z], [reletive_pitch, relative_yaw]])
num_point = len(relative_transform)

def process(image):
    global point
    # save image
    if point:
        image.save_to_disk('output/raw/1-%05d.png' % point)
        image.save_to_disk('output/converted/1-%05d.png' % point, cc.CityScapesPalette)
        print('image 1-%05d.png saved' % point)
    # move to next location and rotation  # transform  [[x,y,z],[pitch,yaw]]
    if point < num_point:
        transform = relative_transform[point]
        sem_camera.set_transform(carla.Transform(carla.Location(x=transform[0][0], y=transform[0][1], z=transform[0][2]), 
                                            carla.Rotation(pitch=transform[1][0], yaw=transform[1][1])))
        point += 1
        time.sleep(1)
    else: sem_camera.stop()


# Create client and connected to the server
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

# get world
world = client.load_world('Town01')
# # setting synchoronous mode
# settings = world.get_settings()
# settings.synchronous_mode = True # If synchronous mode is enabled, and there is a Traffic Manager running, this must be set to sync mode too.
# settings.fixed_delta_seconds = 0.05
# world.apply_settings(settings)
# set weather
weather = carla.WeatherParameters(
    cloudiness=0.5,
    precipitation=0.5,
    sun_altitude_angle=50.0)
world.set_weather(weather)

# get vehicle blueprint
model3_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
model3_bp.set_attribute('color', '255,255,255')
# get camera blueprint
sem_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
sem_bp.set_attribute('image_size_x', '1080')
sem_bp.set_attribute('image_size_y', '720')
# camera_bp.set_attribute('sensor_tick', '0.5')  # Set the time in seconds between sensor captures

# spawn vehicle(Actor)
spawn_points = world.get_map().get_spawn_points()
model3_spawn_point = np.random.choice(spawn_points)
model3 = world.spawn_actor(model3_bp, model3_spawn_point)
# model3.set_autopilot(True)
actor_list.append(model3)

# spawn camera
sem_camera = world.spawn_actor(sem_bp, 
                           carla.Transform(carla.Location(z=5), carla.Rotation(pitch=0)), 
                           model3,
                           carla.AttachmentType.Rigid
                           )
actor_list.append(sem_camera)
sem_camera.listen(lambda image:process(image))


while running:
    if sem_camera.is_listening:
        spectator = world.get_spectator()
        transform = model3.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))
    else:
        for actor in actor_list:
            actor.destroy()
        break


