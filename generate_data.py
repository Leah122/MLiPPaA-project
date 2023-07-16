import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from constants import DETECTORS, DIM, NR_EVENTS, LABEL_FILENAME, DATA_FILENAME
from skspatial.objects import Line, Sphere, Circle
from argparse import ArgumentParser


# defining some constants
DETECTORS = [1,2,3,4,5]
NR_DETECTORS = len(DETECTORS)
DATA_FILENAME = "data/hits.txt"
LABEL_FILENAME = "data/parameters.txt"


# calculates an intersection between a circle and a line
def circle_line_intersection(radius, angle, direction):
    track_vector = angle_to_vector_2d(angle)
    circle = Circle((0, 0), radius)
    line = Line((0, 0), track_vector)
    intersection = circle.intersect_line(line)
    return intersection[direction]


# calculates an intersection between a sphere and a line
def sphere_line_intersection(radius, angle, direction):
    track_vector = angle_to_vector_3d(angle)
    sphere = Sphere([0, 0, 0], radius)
    line = Line([0, 0, 0], track_vector)
    intersection = sphere.intersect_line(line)
    return intersection[direction]


# computes the unit vector (x, y) starting at the origin belonging to a given angle
def angle_to_vector_2d(angle_in_rad):
    x = np.cos(angle_in_rad)
    y = np.sin(angle_in_rad)
    return np.array([x, y])


# computes the unit vector (x, y, z) starting at the origin belonging to the two given angles
def angle_to_vector_3d(angle_in_rad):
    theta_rad, phi_rad = angle_in_rad
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    return np.array([x, y, z])


# adds a circle to the given plot
def plot_circle(center, radius, color='darkgray', alpha=0.7, ax=None):
    circle = plt.Circle(center, radius, color=color, alpha=alpha, fill=False)
    ax.add_artist(circle)


# adds a sphere to the given plot
def plot_sphere(radius, color='darkgray', alpha=0.7, ax=None):
    radius = radius
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha)


# writes the event id, track angles, and track id's to a file
def write_parameters(data, output=LABEL_FILENAME):
    with open(output, "w") as file:
        for key, value in data.items():
            value = value[1::]
            value = ','.join(map(str, value))
            file.write(f"{key},{value}\n")
    print("Successfully written track data")


# writes the event id, hit coordinates, and track id's to a file
def write_data(data, output=DATA_FILENAME):
    with open(output, "w") as file:
        for key, value in data.items():
            value = value[1::]
            value = ','.join(map(str, value))
            file.write(f"{key},{value}\n")
    print("Successfully written parameter data")


# generates the events with the tracks and parameters for 2d data
def generate_data_2d(nr_events=50_000, max_nr_tracks=3, noise=0.1):
    detectors = DETECTORS
    detector_intersect = [0] * (max_nr_tracks * len(detectors))
    
    events = {}
    parameters = {}

    for event in tqdm(range(nr_events)):
        # initialise the events and parameters
        events[event] = [event]
        parameters[event] = [event]

        for track in range(np.random.randint(2, max_nr_tracks)):
            track_angle = random.uniform(-np.pi, np.pi)

            # the line will cross the detector twice, while we only want one, so we choose the direction of the angle
            direction = 1 if track_angle > 0 else 0
            parameters[event].append(track)
            parameters[event].append(track_angle)

            # for each detector we calculate the hit coordinates add some noise, then add them to the event dictionary. 
            for radius in detectors:
                detector_intersect[track] = circle_line_intersection(radius, track_angle, direction)

                events[event].append(detector_intersect[track][0] + np.random.normal(0, noise, 1)[0])
                events[event].append(detector_intersect[track][1] + np.random.normal(0, noise, 1)[0])
                events[event].append(track)
    return events, parameters


# generates the events with the tracks and parameters for 3d data
def generate_data_3d(nr_events=50_000, max_nr_tracks=20, noise=0.1):
    detectors = DETECTORS
    detector_intersect = [0] * (max_nr_tracks * len(detectors))

    events = {}
    parameters = {}

    for event in tqdm(range(nr_events)):
        # add the event id at the beginning of the event
        events[event] = [event]
        parameters[event] = [event]

        for track in range(np.random.randint(2, max_nr_tracks+1)):
            track_angle = [random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)]

            # save the angles as a semi-colon seperated item and append to the parameters
            track_angle_print = f"{track_angle[0]};{track_angle[1]}"
            parameters[event].append(track)
            parameters[event].append(track_angle_print)
            direction = 1 # the direction for sphere intersections is always 1 with this library

            # for each detector, we calculate the hit coordinates add some noise, then add them to the event dictionary
            for radius in detectors:
                detector_intersect[track] = sphere_line_intersection(radius, track_angle, direction)
                events[event].append(detector_intersect[track][0] + np.random.normal(0, noise, 1)[0])
                events[event].append(detector_intersect[track][1] + np.random.normal(0, noise, 1)[0])
                events[event].append(detector_intersect[track][2] + np.random.normal(0, noise, 1)[0])
                events[event].append(track)
    return events, parameters


# makes a plot of the first event that was generated for 2d data
def plot_example_event_2d(data=DATA_FILENAME, parameters=LABEL_FILENAME):
    # initialise figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # plot the circles
    for radius in DETECTORS:
        plot_circle((0, 0), radius, ax=ax)

    # read the angles
    with open(parameters) as f:
        parameter_row = f.readline().split(',')
    parameter_row = [float(i) for i in parameter_row]

    # for every angle, we caluclate the unit vector, make it as big as the biggest detector, then plot it
    for i in range(2, len(parameter_row), 2):
        track_vector = DETECTORS[-1] * angle_to_vector_2d(parameter_row[i])
        plt.plot([0, track_vector[0]], [0, track_vector[1]], alpha=0.6)

    # read the hits
    with open(data) as f:
        data_row = f.readline().split(',')
    data_row = [float(i) for i in data_row]

    # plot the hits
    for i in range(1, len(data_row), 3):
        plt.scatter(data_row[i], data_row[i + 1], marker="o", c="black")

    # set limits and labels, then save the plot to a file
    ax.set_ylim(-6, 6)
    ax.set_xlim(-6, 6)
    plt.grid()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig('figures/example_event.png')


# makes a plot of the first event that was generated for 3d data
def plot_example_event_3d(data=DATA_FILENAME, parameters=LABEL_FILENAME):
    ax = plt.figure(figsize=(7,7)).add_subplot(projection='3d')

    # read the angles
    with open(parameters) as f:
        parameter_row = f.readline().split(',')

    # for every angle, we caluclate the unit vector, make it as big as the biggest detector, then plot it
    for i in range(2, len(parameter_row), 2):
        parameters = parameter_row[i].split(";")
        parameters = [float(i) for i in parameters]
        track_vector = 5 * angle_to_vector_3d([parameters[0], parameters[1]])
        plt.plot([0, track_vector[0]], [0, track_vector[1]], [0, track_vector[2]], alpha=0.6)

    # read the hits
    with open(data) as f:
        data_row = f.readline().split(',')
    data_row = [float(i) for i in data_row]

    # plot the hits
    for i in range(1, len(data_row), 4):
        plt.plot(data_row[i], data_row[i + 1], data_row[i+2], marker="o", c="black")

    # set limits and labels, then save the plot to a file
    ax.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_zlim(-5, 5)
    plt.grid()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    ax.set_zlabel('Z-axis')
    plt.savefig('figures/example_event.png')


if __name__ == '__main__': 
    # set seed for reproducability
    np.random.seed(42)

    # parse the arguments, default is 2d data with 50.000 events
    parser = ArgumentParser()
    parser.add_argument('-d', '--dimensions', type=int,
                        default='2',
                        help='number of dimensions (can be 2 or 3)')
    parser.add_argument('-e', '--nr_events', type=int,
                        default='50000',
                        help='number of events to be generated')
    args = parser.parse_args()
    print('generating ', args.nr_events, ' events with ', args.dimensions, ' dimensions.')
    
    # generate the data
    if args.dimensions == 2:
        data, parameter = generate_data_2d(nr_events=args.nr_events, max_nr_tracks=20)
    elif args.dimensions == 3:
        data, parameter = generate_data_3d(nr_events=args.nr_events, max_nr_tracks=20)
    
    # write the data to files
    write_data(data)
    write_parameters(parameter)

    # plot the first event
    if args.dimensions == 2:
        plot_example_event_2d()
    elif args.dimensions == 3:
        plot_example_event_3d()