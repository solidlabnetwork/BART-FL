import os
import urllib.request
import tarfile

# Define URLs and paths
data_dir = "data"
voc_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
tar_path = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar")
extract_path = data_dir

# Step 1: Create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Step 2: Download VOC tar file if not already downloaded
if not os.path.exists(tar_path):
    print("Downloading VOC 2012 dataset...")
    urllib.request.urlretrieve(voc_url, tar_path)
    print("Download complete.")
else:
    print("VOC tar file already exists. Skipping download.")

# Step 3: Extract tar file
print("Extracting VOC 2012 dataset...")
with tarfile.open(tar_path) as tar:
    tar.extractall(path=extract_path)
print("Extraction complete.")

# Step 4: Check if JPEGImages directory exists
jpeg_dir = os.path.join(data_dir, "VOCdevkit/VOC2012/JPEGImages")
if os.path.exists(jpeg_dir):
    print(f"VOC JPEG images are now available at: {jpeg_dir}")
else:
    print("Something went wrong. JPEGImages directory not found.")
