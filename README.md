# MLiPPaA-project
The data can be generated with the file generate_data.py the usage seen below. If no options are given the default will generate 50.000 events with 2 dimensions. A small sample of the data is provided in the data folder

python generate_data.py -d [nr_dimensions] -s [nr_samples] 

This will output 2 txt files, hits.txt and parameters.txt, containing the hits and parameters respectively as a comma seperated table in the following format:

parameters: event_id,track_id_0,track_parameter_0,track_id_1,track_parameter_1,track_id_2,track_parameter_2,...

hits: event_id,hit_x_0,hit_y_0,track_id_0,hit_x_1,hit_y_1,track_id_1,hit_x_2,hit_y_2,track_id_2,...

If 3d data is generated it will have an addtional z coordinate in the hits and the track parameters will have two parameters seperated by a semicolon.

Training can be done in the notebook, which has instructions within it.
