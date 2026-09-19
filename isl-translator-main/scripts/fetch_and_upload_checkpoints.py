import sys
import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

print("Initializing Kaggle API...")
api = KaggleApi()
api.authenticate()

# Name of the dataset we want to create
DATASET_SLUG = "isl-translator-checkpoints"
DATASET_TITLE = "ISL Translator INCLUDE Checkpoints"
USERNAME = api.get_config_value("username")

# Local directories
OUTPUT_DIR = "temp_kaggle_output"
DATASET_DIR = "isl_checkpoints_dataset"

# Clean up any previous runs
for d in [OUTPUT_DIR, DATASET_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
os.makedirs(OUTPUT_DIR)
os.makedirs(DATASET_DIR)

print("\n🚀 1. Downloading Output from Version 5 of kaggle-include-train")
try:
    # CLI command: kaggle kernels output lastlegend/kaggle-include-train [-p PATH]
    # The API doesn't let us specify a version easily, but pulling the kernel output usually pulls the latest/failed completed.
    # Note: If this just pulls v4, we'll try a different approach. But the user's CLI command in screenshot worked.
    api.kernels_output_cli("lastlegend/kaggle-include-train", path=OUTPUT_DIR)
    print("Download complete!")
except Exception as e:
    print(f"❌ Error downloading: {e}")
    sys.exit(1)

# Check if checkpoints exist in the downloaded output
checkpoints_found = False
for root, dirs, files in os.walk(OUTPUT_DIR):
    if "latest.pt" in files:
        checkpoints_found = True
        print(f"✅ Found latest.pt at: {os.path.join(root, 'latest.pt')}")
        break

if not checkpoints_found:
    print("❌ ERROR: latest.pt was NOT found in the downloaded output.")
    print("This means Kaggle completely deleted it upon timeout. It is unrecoverable.")
    sys.exit(1)

print("\n📦 2. Packaging into a standalone Dataset...")
# Move everything from OUTPUT_DIR into DATASET_DIR
for item in os.listdir(OUTPUT_DIR):
    s = os.path.join(OUTPUT_DIR, item)
    d = os.path.join(DATASET_DIR, item)
    if os.path.isdir(s):
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

# Generate dataset-metadata.json
print("Generating metadata...")
# api.dataset_initialize(DATASET_DIR) # Generates default template
metadata = {
  "title": DATASET_TITLE,
  "id": f"{USERNAME}/{DATASET_SLUG}",
  "licenses": [{"name": "CC0-1.0"}]
}
import json
with open(os.path.join(DATASET_DIR, "dataset-metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\n🚀 3. Uploading Checkpoints as a new Kaggle Dataset...")
try:
    api.dataset_create_new(
        folder=DATASET_DIR, 
        dir_mode="zip",
        public=False,
        quiet=False
    )
    print(f"\n🎉 SUCCESS! Dataset '{DATASET_TITLE}' is currently being created on Kaggle!")
    print(f"URL: https://www.kaggle.com/{USERNAME}/{DATASET_SLUG}")
except Exception as e:
    print(f"❌ Error uploading dataset: {e}")
    sys.exit(1)

print("\nYou can now attach THIS brand new dataset to your notebook instead of the old confusing notebook output!")
