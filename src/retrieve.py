#  Copyright 2023 Custom Diffusion authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from io import BytesIO
from pathlib import Path
import requests
from PIL import Image
from tqdm import tqdm
from serpapi import GoogleSearch
import time

def retrieve(class_prompt, class_data_dir, num_class_images, api_key=None):
    if api_key is None:
        raise ValueError("Please provide a SerpAPI key for image retrieval")
    
    os.makedirs(f"{class_data_dir}/images", exist_ok=True)
    if len(list(Path(f"{class_data_dir}/images").iterdir())) >= num_class_images:
        return

    params = {
        "engine": "google",
        "q": class_prompt,
        "tbm": "isch",
        "num": 100,
        "api_key": api_key
    }

    try:
        print(f"Searching for images with prompt: {class_prompt}")
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if results is None:
            raise Exception("API returned None response")
            
        if "error" in results:
            print(f"API Error: {results['error']}")
            raise Exception(f"API Error: {results['error']}")
            
        if "images_results" not in results:
            print("No images_results in API response")
            print(f"API Response: {results}")
            raise Exception("No images found in API response")
            
        if not results["images_results"]:
            print("Empty images_results list")
            raise Exception("No images found in search results")
            
        print(f"Found {len(results['images_results'])} images")
        
    except Exception as e:
        print(f"Error during search: {str(e)}")
        # Create empty files to prevent training errors
        with open(f"{class_data_dir}/caption.txt", "w") as f1, \
             open(f"{class_data_dir}/urls.txt", "w") as f2, \
             open(f"{class_data_dir}/images.txt", "w") as f3:
            pass
        raise

    count = 0
    pbar = tqdm(desc="downloading real regularization images", total=num_class_images)

    with open(f"{class_data_dir}/caption.txt", "w") as f1, open(f"{class_data_dir}/urls.txt", "w") as f2, open(
        f"{class_data_dir}/images.txt", "w"
    ) as f3:
        for image in results["images_results"]:
            if count >= num_class_images:
                break
                
            try:
                img_url = image.get("original", image.get("link", ""))
                if not img_url:
                    print(f"Skipping image {count}: No valid URL found")
                    continue
                    
                print(f"Downloading image {count + 1} from: {img_url}")
                img = requests.get(img_url, timeout=10)
                if img.status_code == 200:
                    _ = Image.open(BytesIO(img.content))
                    with open(f"{class_data_dir}/images/{count}.jpg", "wb") as f:
                        f.write(img.content)
                    f1.write(class_prompt + "\n")
                    f2.write(img_url + "\n")
                    f3.write(f"{class_data_dir}/images/{count}.jpg" + "\n")
                    count += 1
                    pbar.update(1)
                    time.sleep(1)
                else:
                    print(f"Skipping image {count}: HTTP status {img.status_code}")
            except Exception as e:
                print(f"Error downloading image {count}: {str(e)}")
                continue

    if count == 0:
        # Create empty files to prevent training errors
        with open(f"{class_data_dir}/caption.txt", "w") as f1, \
             open(f"{class_data_dir}/urls.txt", "w") as f2, \
             open(f"{class_data_dir}/images.txt", "w") as f3:
            pass
        raise Exception("Failed to download any images")
        
    return

def parse_args():
    parser = argparse.ArgumentParser("", add_help=False)
    parser.add_argument("--class_prompt", help="text prompt to retrieve images", required=True, type=str)
    parser.add_argument("--class_data_dir", help="path to save images", required=True, type=str)
    parser.add_argument("--num_class_images", help="number of images to download", default=200, type=int)
    parser.add_argument("--api_key", help="SerpAPI key for image retrieval", required=True, type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    retrieve(args.class_prompt, args.class_data_dir, args.num_class_images,args.api_key)
