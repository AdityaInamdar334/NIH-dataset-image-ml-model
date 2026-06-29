import os
import urllib.request
from pathlib import Path
import sys
import ssl

# Fix SSL certificate verify failed error on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Official NIH Box URLs for the 12 image archives
# Total size is ~42 GB.
IMAGE_LINKS = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]

# A reliable mirror for the CSV file (HuggingFace datasets mirror of the NIH dataset)
CSV_URL = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/Data_Entry_2017_v2020.csv"


def download_progress_hook(block_num, block_size, total_size):
    """Callback to print download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        sys.stdout.write(f"\rDownloaded {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({percent:.1f}%)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloaded {downloaded / (1024*1024):.2f} MB")
        sys.stdout.flush()


def download_dataset(download_all: bool = False):
    """
    Downloads the Data_Entry_2017.csv and the image tar.gz files.
    If download_all is False, it will only download the first image archive (~2GB) to save time for prototyping.
    """
    data_dir = Path('./data')
    archives_dir = data_dir / 'archives'
    
    data_dir.mkdir(exist_ok=True)
    archives_dir.mkdir(exist_ok=True)

    # 1. Download CSV
    csv_path = data_dir / 'Data_Entry_2017.csv'
    if not csv_path.exists():
        print(f"Downloading CSV metadata to {csv_path}...")
        try:
            urllib.request.urlretrieve(CSV_URL, csv_path)
            print("\nCSV downloaded successfully!")
        except Exception as e:
            print(f"\nFailed to download CSV: {e}")
            return
    else:
        print(f"CSV already exists at {csv_path}")

    # 2. Download Image Archives
    links_to_download = IMAGE_LINKS if download_all else IMAGE_LINKS[:1]
    
    print(f"\nPreparing to download {len(links_to_download)} image archive(s)...")
    
    for idx, link in enumerate(links_to_download):
        filename = f'images_{idx+1:02d}.tar.gz'
        filepath = archives_dir / filename
        
        if filepath.exists():
            print(f"{filename} already exists, skipping.")
            continue
            
        print(f"\nDownloading {filename}...")
        try:
            urllib.request.urlretrieve(link, filepath, reporthook=download_progress_hook)
            print(f"\nSuccessfully downloaded {filename}")
        except Exception as e:
            print(f"\nFailed to download {filename}: {e}")

    print("\n--- Download Phase Complete ---")
    if not download_all:
        print("Note: Only downloaded images_01.tar.gz (~5,000 images) for rapid prototyping.")
        print("To download all 112,000 images, change `download_all=True` in this script.")


if __name__ == '__main__':
    # Set this to True if you want to wait for all ~42GB to download.
    # Leaving it False will download ~2GB (images_01) which is perfect for testing!
    DOWNLOAD_ALL_IMAGES = True
    download_dataset(DOWNLOAD_ALL_IMAGES)
