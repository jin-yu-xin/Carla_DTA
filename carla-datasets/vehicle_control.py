"""
Use WASD keys for control.
    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
"""

import glob
import os
import sys
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
try:
    import pygame
    from pygame.locals import K_w
    from pygame.locals import K_s
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_q
    from pygame.locals import K_SPACE
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


actor_list = []
steer_cache = 0.0
vehicle_controller = carla.VehicleControl()

def parse_control_key(keys, milliseconds):
    if keys[K_q]:
        vehicle_controller.gear = 1 if vehicle_controller.reverse else -1

    if keys[K_w]:
        vehicle_controller.throttle = min(vehicle_controller.throttle + 0.01, 1.00)
    else:
        vehicle_controller.throttle = 0.0

    if keys[K_s]:
        vehicle_controller.brake = min(vehicle_controller.brake + 0.2, 1)
    else:
        vehicle_controller.brake = 0
    
    steer_increment = 5e-4 *  milliseconds
    if keys[K_a]:  # turn left 
        if steer_cache > 0:
            steer_cache = 0
        else:
            steer_cache -= steer_increment
    elif keys[K_d]:  # turn right
        if steer_cache < 0:
            steer_cache = 0
        else:
            steer_cache += steer_increment
    else:
        steer_cache = 0
    steer_cache = min(0.7, max(-0.7, steer_cache))
    vehicle_controller.steer = round(steer_cache, 1)
    vehicle_controller.hand_brake = keys[K_SPACE]

    vehicle_controller.reverse = True if vehicle_controller.gear < 0 else False
    return vehicle_controller


def game_loop():
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        # Create client and connected to the server
        client = carla.Client("localhost", 2000)
        client.set_timeout(2.0)
        # get world
        world = client.load_world('Town03')
        original_settings = world.get_settings()
        # # We set CARLA syncronous mode
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        # get vehicle blueprint
        model3_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        model3_bp.set_attribute('color', '128,128,128')
        # spawn vehicle(Actor)
        spawn_points = world.get_map().get_spawn_points()
        model3_spawn_point = np.random.choice(spawn_points)
        model3 = world.spawn_actor(model3_bp, model3_spawn_point)
        actor_list.append(model3)
        
        clock = pygame.time.Clock()

        while True:
            world.tick()
            clock.tick_busy_loop(60)
            controller = parse_control_key(pygame.key.get_pressed(), clock.get_time())
            model3.apply_control(vehicle_controller)
    finally:
        if original_settings:
            world.apply_settings(original_settings)
        pygame.quit()


def main():
    try:
        game_loop()
    except KeyboardInterrupt:
        print('\nCancelled by user!')


if __name__ == "__main__":
    main()
