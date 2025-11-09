import kagglehub
import shutil
import os

# Define the target path in your project repository
project_repo_path = "/disk/solidlab-server/lclhome/mmia001/bart-fl/data"

# Download the latest version of the dataset
path = kagglehub.dataset_download("chandanakuntala/cropped-lisa-traffic-light-dataset")




# Copy dataset to the project repository
if not os.path.exists(project_repo_path):
   os.makedirs(project_repo_path)




# Copy downloaded dataset to the project repo
shutil.copytree(path, project_repo_path, dirs_exist_ok=True)




print("Dataset saved to project repository at:", project_repo_path)



