import os
import pickle
import scipy.io

# Specify the directory containing the .pkl files
directory = "Evaluation/OfflineEvalResults/MidSIR"
# Initialize a dictionary to hold all the data
data_dict = {}

# Loop over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .pkl file
    if filename.endswith('.pkl'):
        # Open the .pkl file and load its data
        with open(os.path.join(directory, filename), 'rb') as f:
            data = pickle.load(f)
        
        # Add the data to the dictionary, using the filename (without .pkl) as the key
        data_dict[filename[:-4]] = data

# Save the dictionary to a .mat file
folder_name = 'X_Mat_Results'
os.makedirs(folder_name, exist_ok=True)
scipy.io.savemat(folder_name + '/MidSIR.mat', data_dict)



# # Load the .mat file
# mat_contents = scipy.io.loadmat('X_Mat_Results/GeneralTest/combined.mat')

# # mat_contents is now a dictionary with variable names as keys
# # and loaded matrices as values. Let's print the keys (variable names)
# for name in mat_contents.keys():
#     print(name)
# print(len(mat_contents))