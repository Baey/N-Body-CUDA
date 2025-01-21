import numpy as np
from vispy import scene, app
import os
import argparse
from utils import read_initial_conditions, read_trajectories
import seaborn as sns
import random

# Parse command-line arguments
parser = argparse.ArgumentParser(description="N-Body Visualization")
parser.add_argument("initial_conditions", help="Path to the initial conditions file")
parser.add_argument("trajectories", help="Path to the trajectories file")
args = parser.parse_args()

# Read initial conditions and trajectories
bodies = read_initial_conditions(args.initial_conditions)
trajectories = read_trajectories(args.trajectories)

# Create VisPy scene
canvas = scene.SceneCanvas(keys="interactive", bgcolor="black")
view = canvas.central_widget.add_view()
camera = scene.TurntableCamera(fov=60)
view.camera = camera

# Get a color palette from seaborn and shuffle it with a fixed random seed
random_state = 41
palette = sns.color_palette("husl", len(bodies))
random.Random(random_state).shuffle(palette)
colors = [tuple(color) + (1,) for color in palette]  # Add alpha value

# Create 3D spheres for each body
spheres = []
trails = []
for i, body in enumerate(bodies):
    sphere = scene.visuals.Sphere(
        radius=body["radius"],
        color=colors[i],
        method="latitude",  # Use latitude-longitude method for better quality
        subdivisions=64,  # Increase the number of segments for better quality
    )
    sphere.transform = scene.transforms.MatrixTransform()
    sphere.transform.translate(body["position"])  # Scale position for visualization
    view.add(sphere)
    spheres.append(sphere)

    trail = scene.visuals.Line(color=colors[i], width=1, antialias=True)
    view.add(trail)
    trails.append(trail)

# Animation
step = 0
trail_length = 10000  # Number of steps to keep in the trail
current_body = 0  # Index of the current body to focus on
paused = False  # Simulation paused state
speed_factor = 1  # Speed factor for the simulation


def update(event):
    global step
    if paused:
        return

    step += speed_factor
    if step >= len(next(iter(trajectories.values()))):
        step = 0  # Loop the animation

    for i, (sphere, trail) in enumerate(zip(spheres, trails)):
        position = trajectories[i][step]  # Scale position for visualization
        sphere.transform.reset()
        sphere.transform.translate(position)

        # Update trail with fading effect
        trail_data = np.array(trajectories[i][max(0, step - trail_length) : step + 1])
        alphas = np.linspace(0, 1, len(trail_data))  # Create fading effect
        colors_with_alpha = np.array([colors[i][:3] + (alpha,) for alpha in alphas])
        trail.set_data(trail_data, color=colors_with_alpha, connect="strip")

    # Update camera center
    camera.center = trajectories[current_body][step]


def on_key_press(event):
    global current_body, paused, speed_factor
    if event.key == "N":
        current_body = (current_body + 1) % len(bodies)
    elif event.key == "P":
        current_body = (current_body - 1) % len(bodies)
    elif event.key == "Space":
        paused = not paused
    elif event.key == "Up":
        speed_factor = min(speed_factor + 1, 100000)  # Increase speed, max 10x
    elif event.key == "Down":
        speed_factor = max(speed_factor - 1, 1)  # Decrease speed, min 1x


canvas.events.key_press.connect(on_key_press)
timer = app.Timer(interval=1 / 60, connect=update, start=True)

# Show canvas
canvas.show()
app.run()
