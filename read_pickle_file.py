import pandas as pd
import pickle

# Replace 'path_to_your_pickle_file.pkl' with the actual path to your pickle file
pickle_file_path = '/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle'

# Load your DataFrame from the pickle file
df = pd.read_pickle(pickle_file_path)

# Set the option to display all columns of the DataFrame
pd.set_option('display.max_columns', None)

# Print the DataFrame to show all columns
print(df)
