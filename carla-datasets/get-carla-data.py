import glob
import os
import sys
import math
from queue import Queue
from queue import Empty
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
num_point = len(relative_transform) # 72


# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))
    
def main():
    global point
    # Create client and connected to the server
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    # get world
    world = client.load_world('Town01')

    try:
        original_settings = world.get_settings()
        # We set CARLA syncronous mode
        settings = world.get_settings()
        settings.fixed_delta_seconds = 1
        settings.synchronous_mode = True
        world.apply_settings(settings)
        # set weather
        weather = carla.WeatherParameters(
            cloudiness=35,
            precipitation=0,
            sun_altitude_angle=45.0)
        world.set_weather(weather)


        # ######################################## vehicle bluprints ###############################################################################
        # get vehicle blueprint
        model3_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        color_list = model3_bp.get_attribute('color').recommended_values
        print(color_list)
        model3_bp.set_attribute('color', '128,128,128')
        ########################################## spawn vehicle ###################################################################################
        spawn_points = world.get_map().get_spawn_points()
        model3_spawn_point = np.random.choice(spawn_points, 5, replace=False)

        for i, spawn_point in enumerate(model3_spawn_point):
            flag = 0
            actor_list = []
            sensor_list = []
            model3_bp.set_attribute('color', '128,128,128')
            model3 = world.spawn_actor(model3_bp, spawn_point)
            # model3.set_autopilot(True)
            actor_list.append(model3)

            spectator = world.get_spectator()
            transform = model3.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=15), carla.Rotation(pitch=-90)))
            ############################################ camera sensor blueprint ########################################################################
            # sensor queue in which we keep track of the informationalready received. 
            # This structure is thread safe and can be accessed by all the sensors callback concurrently without problem.
            sensor_queue = Queue()
            # rgb camera
            rgb_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_cam_bp.set_attribute('image_size_x', '1080')
            rgb_cam_bp.set_attribute('image_size_y', '720')
            # semantic segmentation camera
            sem_cam_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            sem_cam_bp.set_attribute('image_size_x', '1080')
            sem_cam_bp.set_attribute('image_size_y', '720')

            ######################################### spawn camera #####################################################################################
            spawn_transform = relative_transform[0]
            rgb_cam = world.spawn_actor(rgb_cam_bp, 
                                    carla.Transform(carla.Location(x=spawn_transform[0][0], y=spawn_transform[0][1], z=spawn_transform[0][2]), 
                                                        carla.Rotation(pitch=spawn_transform[1][0], yaw=spawn_transform[1][1])), 
                                    model3,
                                    carla.AttachmentType.Rigid
                                    )
            rgb_cam.listen(lambda data:sensor_callback(data, sensor_queue, "rgb-camera"))  # Receive data on every tick
            sensor_list.append(rgb_cam)
            actor_list.append(rgb_cam)
            # spawn  semantic segmentation camera
            sem_cam = world.spawn_actor(sem_cam_bp, 
                                    carla.Transform(carla.Location(x=spawn_transform[0][0], y=spawn_transform[0][1], z=spawn_transform[0][2]), 
                                                        carla.Rotation(pitch=spawn_transform[1][0], yaw=spawn_transform[1][1])), 
                                    model3,
                                    carla.AttachmentType.Rigid
                                    )
            sem_cam.listen(lambda data:sensor_callback(data, sensor_queue, "sem-camera"))  # Receive data on every tick
            sensor_list.append(sem_cam)
            actor_list.append(sem_cam)

            point = 0
            while running:
                world.tick()
                if flag == 0:
                    w_frame = world.get_snapshot().frame
                    print("\nWorld's frame: %d" % w_frame)
                    for _ in range(len(sensor_list)):
                        s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                        print('do not save the first %s frame' % s_name)
                    flag = 1
                    continue
                if point < num_point:
                    w_frame = world.get_snapshot().frame
                    print("\nWorld's frame: %d" % w_frame)

                    try:
                        for _ in range(len(sensor_list)):
                            s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                            print("    Frame: %d   Sensor: %s" % (s_frame, s_name))
                            # save image
                            if point < num_point and s_name == "rgb-camera":
                                s_data.save_to_disk('output18/raw/%03d-000-%03d.png' % (i, point), cc.Raw)  # 32-bit BGRA colors
                                print('    rgb-image %03d-000-%03d.png saved' % (i, point))
                            if point < num_point and s_name == "sem-camera": 
                                s_data.save_to_disk('output18/converted/%03d-000-%03d.png' % (i, point), cc.CityScapesPalette)
                                print('    sem-image %03d-000-%03d.png saved' % (i, point))
                    except Empty:
                        print("    Some of the sensor information is missed")
                    
                    # move to next transform
                    point += 1
                    if point < num_point:
                        transform = relative_transform[point]
                        rgb_cam.set_transform(carla.Transform(carla.Location(x=transform[0][0], y=transform[0][1], z=transform[0][2]), 
                                                                carla.Rotation(pitch=transform[1][0], yaw=transform[1][1])))
                        sem_cam.set_transform(carla.Transform(carla.Location(x=transform[0][0], y=transform[0][1], z=transform[0][2]), 
                                                                carla.Rotation(pitch=transform[1][0], yaw=transform[1][1])))
                else:
                    # rgb_cam.stop()
                    # sem_cam.stop()
                    for actor in actor_list:
                        actor.destroy()
                    print('all actors destroyed!!!')
                    break
                time.sleep(1)
                flag = 0

            for j, color in enumerate(color_list):
                j += 1
                flag = 0
                actor_list = []
                sensor_list = []
                model3_bp.set_attribute('color', color)
                model3 = world.spawn_actor(model3_bp, spawn_point)
                # model3.set_autopilot(True)
                actor_list.append(model3)

                spectator = world.get_spectator()
                transform = model3.get_transform()
                spectator.set_transform(carla.Transform(transform.location + carla.Location(z=15), carla.Rotation(pitch=-90)))
                ############################################ camera sensor blueprint ########################################################################
                # sensor queue in which we keep track of the informationalready received. 
                # This structure is thread safe and can be accessed by all the sensors callback concurrently without problem.
                sensor_queue = Queue()
                # rgb camera
                rgb_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
                rgb_cam_bp.set_attribute('image_size_x', '1080')
                rgb_cam_bp.set_attribute('image_size_y', '720')

                ######################################### spawn camera #####################################################################################
                spawn_transform = relative_transform[0]
                rgb_cam = world.spawn_actor(rgb_cam_bp, 
                                        carla.Transform(carla.Location(x=spawn_transform[0][0], y=spawn_transform[0][1], z=spawn_transform[0][2]), 
                                                            carla.Rotation(pitch=spawn_transform[1][0], yaw=spawn_transform[1][1])), 
                                        model3,
                                        carla.AttachmentType.Rigid
                                        )
                rgb_cam.listen(lambda data:sensor_callback(data, sensor_queue, "rgb-camera"))  # Receive data on every tick
                sensor_list.append(rgb_cam)
                actor_list.append(rgb_cam)

                point = 0
                while running:
                    world.tick()
                    if flag == 0:
                        w_frame = world.get_snapshot().frame
                        print("\nWorld's frame: %d" % w_frame)
                        for _ in range(len(sensor_list)):
                            s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                            print('do not save the first %s frame' % s_name)
                        flag = 1
                        continue
                    if point < num_point:
                        w_frame = world.get_snapshot().frame
                        print("\nWorld's frame: %d" % w_frame)

                        try:
                            for _ in range(len(sensor_list)):
                                s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                                print("    Frame: %d   Sensor: %s" % (s_frame, s_name))
                                # save image
                                if point < num_point and s_name == "rgb-camera":
                                    s_data.save_to_disk('output18/rendered/%03d-%03d-%03d.png' % (i, j, point), cc.Raw)  # 32-bit BGRA colors
                                    print('    rgb-image %03d-%03d-%03d.png saved' % (i, j, point))
                        except Empty:
                            print("    Some of the sensor information is missed")
                        
                        # move to next transform
                        point += 1
                        if point < num_point:
                            transform = relative_transform[point]
                            rgb_cam.set_transform(carla.Transform(carla.Location(x=transform[0][0], y=transform[0][1], z=transform[0][2]), 
                                                                    carla.Rotation(pitch=transform[1][0], yaw=transform[1][1])))
                    else:
                        # rgb_cam.stop()
                        # sem_cam.stop()
                        for actor in actor_list:
                            actor.destroy()
                        print('all actors destroyed!!!')
                        break
                    time.sleep(1)
                    flag = 0


    finally:
        world.apply_settings(original_settings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')


