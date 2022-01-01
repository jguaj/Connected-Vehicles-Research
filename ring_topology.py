"""
This module is dedicated to handling data from traffic-simulation.de/ring.html
A general overview goes as follows:
1. Import the dataset from the provided .txt file
2. Convert position data to x/y coordinates
3. Calculates k nearest neighbors for each vehicle at each point in time
4. Randomly assigns each node as honest/dishonest using specified ratio parameter
5. Skews dishonest data using specified standard deviation parameter
6. Calculated integrity of specified vehicle, from specified vehicle's perspective, across specified timeframe

"""

import random
import pandas as pd
import numpy as np
import plotly.express as px
from math import radians, cos, sin, pow, e
from sklearn.neighbors import BallTree
import networkx as nx

np.random.seed(538)
random.seed(231)

def import_from_txt(txt_file_path):
    """
    # Imports given text file argument as dataframe
    Renames dataframe columns to be easier to work with and more descriptive
    x[m] is renamed to pos. It indicates distance around the circumference of the circle
    y[m] is renamed to lane. It indicates which lane the node is in

    :param file_path: requires a path to a txt file as its argument
    :return:  Vehicle dataset with modifications as stated above
    """
    # Importing dataframe, accounting for space delimiters as present in the .txt dataset retrieved from traffic-simulation.de
    vehicle_df = pd.read_csv(txt_file_path, sep='[\s]{1,}', engine='python') 

    vehicle_df.columns = ['time', 'id', 'pos', 'lane', 'speed', 'heading', 'acc']  

    return vehicle_df

def convert_pos_to_xy_pos(vehicle_df):

    """
    # Converts position values into cartesian x/y coordinates
    1. Converts the pos values to radians
    2. Uses sine/cosine to convert radians to relative length
    3. Calculates radius, accounting for which lane the car is in
    4. Multiplies radius by relative length to find the x/y coordinates
    :param ring_vehicle_df: Requires vehicle_df reference variable as input
    :return: No return. vehicle_df is assigned directly by the function
    """
    pi = 3.14159
    circumference = vehicle_df['pos'].max()
    radius = circumference / (2 * pi)

    # Conversion factor used to convert position into radians
    conversion_factor = 1 / (circumference / 360) # * (pi / 180)

    x_positions = []
    y_positions = []

    for idx, row in vehicle_df.iterrows():
        # Converting the position to radians
        pos_radians = radians(row['pos'] * conversion_factor)

        # Adding a 10 unit offset between lanes
        lane_offset = row['lane'] * 10

        # Converting radians into x/y positions and appending them to corresponding lists
        x_positions.append(round(((radius + lane_offset) * cos(pos_radians)), 2))
        y_positions.append(round(((radius + lane_offset) * sin(pos_radians)), 2))

    # Adding x/y position columns to the dataframe
    vehicle_df['xpos'] = x_positions
    vehicle_df['ypos'] = y_positions

    # Dropping the original positions column from the dataframe
    #vehicle_df.drop('pos', axis=1, inplace=True)

def assign_honesty_booleans(vehicle_df, ratio_of_honest_to_dishonest_nodes):
    ratio = ratio_of_honest_to_dishonest_nodes

    id_honesty = {}
    for cur_id in vehicle_df["id"]:
        if cur_id not in id_honesty:
            id_honesty[cur_id] = random.randrange(100) < ratio

    vehicle_df["honesty"] = [id_honesty[i] for i in vehicle_df.id]

    return id_honesty

def plot_animated_scatterplot(vehicle_df):
    """
    :param ring_vehicle_df: In this dataframe, the position data has been replaced with circular x/y coordinates
    :return: Method has no return. Opens animated scatter plot in preferred browser tab
    """

     # Sorting the dataframe by id and then time in ascending order. This is necessary so that the animation iterates properly
    vehicle_df.sort_values(['id', 'time'], ascending=[True, True], inplace=True)
    # Resetting dataframe index to account for the sort function
    vehicle_df.reset_index(drop=True, inplace=True)

    # This is calling the scatter function from Plotly Express to define the fig variable
    fig = px.scatter(
        vehicle_df,  # The entire dataset is passed as a parameter
        x="xpos",  # x is defined per entries in the xpos column
        y="ypos",  # y is defined per entries in the ypos column
        animation_frame="time",  # Time is the metric by which the animation iterates frames
        color="honesty",  # The nodes are denoted in color by their associated honesty boolean
        hover_name="id",  # When you hover over the node in the animation you will see its id
        range_x=[-200, 200],  # Dimensions of the x axis for the plot
        range_y=[-200, 200]  # Dimensions of the y axis for the plot
    )
    # Rendering the established figure
    fig.show()

def calculate_nearest_neighbors(vehicle_df, k_nearest_neighbors):
    """
    For each time in simulation timespan
        For each unique vehicle_id
            Computes its k nearest neighbors using BallTree algorithm

    vehicle_df is left with k added columns containing the nearest neighbors for every vehicle at every point in time    
    """

    for NN in range(k_nearest_neighbors):
        vehicle_df[f'NN_{NN + 1}'] = ''

    maxtime, mintime = int(vehicle_df['time'].max()), int(vehicle_df['time'].min())

    for time in range(mintime, maxtime + 1):
        XY_positions = vehicle_df[vehicle_df['time'] == time][['xpos', 'ypos','id']]
        XY_positions.reset_index(drop=True, inplace=True)
        
        for i in range(len(XY_positions)): 

            TEMP = XY_positions['id'] 
            XY_positions.drop('id', axis=1, inplace=True) 
            
            tree = BallTree(XY_positions, leaf_size=2)
            distances, indices = tree.query(XY_positions[i:i+1], k=k_nearest_neighbors + 1)
            indices = np.delete(indices[0], 0) # Removing the first nearest neighbor from the indices because the closest node to any node will be the node itself
            
            XY_positions['id'] = TEMP 
            
            cur_id = XY_positions.loc[i, 'id']
            cur_df_index = vehicle_df.loc[(vehicle_df['time']==time) & (vehicle_df['id'] == cur_id)].index.values[0]
            for NN in range (k_nearest_neighbors):
                cur_col_label = 'NN_%d' % (NN + 1) 
                vehicle_df.at[cur_df_index, cur_col_label] = XY_positions.loc[indices[NN], 'id']
        print("NN Computed For Time %f" % time)

def skew_dishonest_vehicle_data(vehicle_df, standard_deviation_in_meters):

    dishonest_vehicle_df = vehicle_df.copy()

    std = standard_deviation_in_meters
    noise = np.random.normal(0,std,len(dishonest_vehicle_df.index))

    dishonest_vehicle_df['xpos'] = vehicle_df['xpos'] + noise
            
    return dishonest_vehicle_df

def calculate_benevolence(position_matrix, honest_data, dishonest_data, primary_observer_id, primary_subject_id, are_observers_honest):
    
    for observer_index, matrix_row in position_matrix.iterrows():
            for subject_index, value in position_matrix.iteritems():
                if are_observers_honest[observer_index]:
                    matrix_row[subject_index] = honest_data.loc[subject_index, 'xpos']
                else:
                    matrix_row[subject_index] = dishonest_data.loc[subject_index, 'xpos']
        
            
    primary_observers_view_of_primary_subject =  position_matrix.loc[primary_observer_id].at[primary_subject_id]
    position_matrix.loc[primary_observer_id].at[primary_subject_id] = None # Removing the observer's data so it is not counted twice
          
    average_of_other_observations_for_primary_subject = position_matrix[primary_subject_id].mean() 
    delta = primary_observers_view_of_primary_subject - average_of_other_observations_for_primary_subject
    sigma = pow(delta, 2)
    benevolence = pow(e, -1 * sigma)

    return benevolence

def calculate_integrity(honest_df, dishonest_df, primary_observer_id, primary_subject_id, start_time, end_time):
    df = honest_df[(honest_df['time']==start_time)] # Limiting dataframe to the start time so there aren't duplicate id's in subsequent commands

    vehicle_ids = df.loc[honest_df['id'] == primary_observer_id].iloc[:, 10:].values[0].tolist() # NOTE: Nearest Neighbor Column Indices are hardcoded to improve readability. This shouldn't be an issue unless main method calls are rearranged
    vehicle_ids.append(primary_observer_id) # Adding the primary observer itself into the list

    are_observers_honest = dict(zip(df.id, df.honesty)) # Creating a dictionary mapping vehicle_ids with their associated honesties

    position_matrix = pd.DataFrame('', columns=vehicle_ids, index=vehicle_ids) #Each row holds a vehicle and the positions it reports for all other vehicles, per its honesty

    positive_outcomes = 0
    total_outcomes = 0

    for time in range(start_time, end_time):
        current_honest_df = honest_df[(honest_df['time']==time)].set_index('id') # Limiting dataframe to current timeframe
        current_dishonest_df = dishonest_df[(honest_df['time']==time)].set_index('id') # Limiting dataframe to current timeframe

        benevolence = calculate_benevolence(position_matrix, current_honest_df, current_dishonest_df, primary_observer_id, primary_subject_id, are_observers_honest)
        
        if benevolence > 0.5:
            positive_outcomes = positive_outcomes + 1

        total_outcomes = total_outcomes + 1
        
    integrity = (positive_outcomes / total_outcomes)
    print(integrity)

#vehicle_df = import_from_txt(r"C:\Users\johnw\Downloads\road1_time351.2.txt")
#convert_pos_to_xy_pos(vehicle_df)
#assign_honesty_booleans(vehicle_df, ratio_of_honest_to_dishonest_nodes=50)
#calculate_nearest_neighbors(vehicle_df, k_nearest_neighbors=10)
#vehicle_df.to_csv(r"C:\Users\johnw\Downloads\351.2_NearestNeighbors.txt", index=False)
vehicle_df = pd.read_csv(r"C:\Users\johnw\Downloads\351.2_NearestNeighbors.txt")
dishonest_vehicle_df = skew_dishonest_vehicle_data(vehicle_df, standard_deviation_in_meters=1.5)
calculate_integrity(vehicle_df, dishonest_vehicle_df, primary_observer_id=215, primary_subject_id=221, start_time=252, end_time=300)

#plot_animated_scatterplot(vehicle_df) 