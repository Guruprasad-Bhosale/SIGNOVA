import os
import requests
import zipfile
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_file(url, target_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as file, tqdm(
        desc=target_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024*1024):
            size = file.write(data)
            bar.update(size)

def main():
    record_id = "4010759"
    api_url = f"https://zenodo.org/api/records/{record_id}"
    
    raw_dir = Path("data/raw/include")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    target_zips = [
        'Greetings_1of2.zip', 
        'Greetings_2of2.zip', 
        'Colours_1of2.zip', 
        'Colours_2of2.zip'
    ]
    
    logger.info(f"Fetching metadata from {api_url}...")
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Failed to fetch metadata: {e}")
        return

    files = data.get('files', [])
    target_files = [(f['key'], f['links']['self']) for f in files if f['key'] in target_zips]

    logger.info(f"Downloading subset: {len(target_files)} relevant files...")
    
    for filename, download_url in target_files:
        filepath = raw_dir / filename
        if filepath.exists():
            logger.info(f"Skipping {filename}, already exists.")
            continue
            
        logger.info(f"\nDownloading {filename}...")
        try:
            download_file(download_url, filepath)
            logger.info(f"Extracting {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            filepath.unlink()
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            continue

    logger.info("INCLUDE subset download & extraction complete.")

if __name__ == "__main__":
    main()
