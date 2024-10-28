import os
import sys
import shutil
import random
import json
from typing import List, Dict, Optional

# Application modules
sys.path.insert(0, os.path.abspath('../'))
from utils.iotools import read_json
from tqdm import tqdm as tqdm

class SplitCUHKPEDES:
   
    def __init__(self, dataset_path: str = None, split_type: str = 'test', category_names: List[str] = None, seed: Optional[int] = None):
        """
        Doc.: CUHK-PEDES dataset from Person Search with Natural Language Description (https://arxiv.org/pdf/1702.05729.pdf)

        Dataset Statistics
        -------------------
        • train split:  34,054 images and 68,126 descriptions for 11,003 persons (ID: 1-11003)
        • val split:    3,078  images and 6,158 descriptions for 1,000 persons (ID: 11004-12003)
        • test split:   3,074  images and 6,156 descriptions for 1,000 persons (ID: 12004-13003)

        Totals:
        -------------------
        • images: 40,206
        • persons: 13,003
                
        Annotation format: 
        [{'split', str,
        'captions', list,
        'file_path', str,
        'processed_tokens', list,
        'id', int}...]
        """
        self.dataset_dir = dataset_path
        self.gallery_dir = os.path.join("..","data","gallery")
        self.img_dir = os.path.join(self.dataset_dir, 'imgs')
        self.anno_path = os.path.join(self.dataset_dir, 'reid_raw.json')
        self.category_names = category_names if category_names else ['cameraone', 'cameratwo', 'camerathree', 'camerafour', 'camerafive']
        self.seed = seed
        self.stats = {category: 0 for category in self.category_names}
        self.stats['all_gallery'] = 0

        self._check_before_run()
        self.train_annos, self.test_annos, self.val_annos = self._split_anno()

        if split_type == 'val':
            self.data = self.filter_unique_ids_and_captions(self.val_annos)
        elif split_type == 'test':
            self.data = self.filter_unique_ids_and_captions(self.test_annos)
        else:
            raise ValueError(f"Invalid value for split_type: {split_type}")
        
        

    def _split_anno(self):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(self.anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

    def filter_unique_ids_and_captions(self, dict_list: List[Dict]) -> List[Dict]:
        """
        Filters a list of dictionaries to ensure only one unique instance of each ID
        and selects only one caption from the list of captions. Does not add dictionaries
        with empty captions to the filtered list.
        
        Parameters:
            dict_list (list): List of dictionaries to filter. Each dictionary must contain an 'id' key and a 'captions' key which is a list.
        
        Returns:
            list: A list of dictionaries with unique IDs and a single caption.
        """
        unique_ids = set()
        filtered_list = []
        
        for item in dict_list:
            if item['id'] not in unique_ids and item['captions']:
                unique_ids.add(item['id'])
                single_caption_item = item.copy()
                single_caption_item['captions'] = [item['captions'][0]]
                filtered_list.append(single_caption_item)
        
        return filtered_list

    def split_and_copy_images(self) -> List[Dict]:
        """
        Splits the self.data into n categories and copies images to corresponding directories.
        
        Returns:
            list: A list of dictionaries with the new key 'moved_to_directory'.
        """
        # Set the random seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)


        # Create directories
        for category in (self.category_names + ['all_gallery']):
            category_path = os.path.join(self.gallery_dir, category)
            if os.path.exists(category_path):
                shutil.rmtree(category_path)
            os.makedirs(category_path, exist_ok=True)

        # Shuffle and split data
        random.shuffle(self.data)
        split_data = [[] for _ in range(len(self.category_names))]
        for i, item in enumerate(self.data):
            split_data[i % len(self.category_names)].append(item)

        

        global_list = []
        for idx, category_data in enumerate(split_data, start=0):
            category_dir = self.category_names[idx]
            for item in category_data:
                # Copy image to category directory
                src = os.path.join(self.img_dir, item['file_path'])
                dst = os.path.join(self.gallery_dir, category_dir, os.path.basename(src))
                if 'processed_tokens' in item:
                    del item['processed_tokens']
                item['moved_to_directory'] = dst
                shutil.copy(src, dst)
                global_list.append(item)

                self.stats[category_dir] += 1

                # Copy image to all_gallery directory
                dst_all_gallery = os.path.join(self.gallery_dir, "all_gallery", os.path.basename(src))
                shutil.copy(src, dst_all_gallery)
                self.stats['all_gallery'] += 1

        return global_list

    def save_global_list_to_file(self, global_list):
        """
        Saves the global list to a text file named global_file_list.txt.
        """
        file_path = os.path.join(self.gallery_dir, 'global_file_list.txt')
        with open(file_path, 'w') as file:
            json.dump(global_list, file, indent=4)

    def print_statistics(self):
        """
        Prints the statistics of how many images were copied into each category.
        """
        print("Statistics of copied images:")
        for category, count in self.stats.items():
            print(f"{category}: {count} images")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not os.path.exists(self.img_dir):
            raise RuntimeError(f"'{self.img_dir}' is not available")
        if not os.path.exists(self.anno_path):
            raise RuntimeError(f"'{self.anno_path}' is not available")


if __name__ == "__main__":
    category_names = ['cameraone', 'cameratwo', 'camerathree', 'camerafour', 'camerafive', 'camerasix', \
                    'cameraseven', 'cameraeight', 'cameranine', 'cameraten', 'cameraeleven', 'cameratwelve', \
                    'camerathirteen', 'camerafourteen', 'camerafifteen', 'camerasixteen', 'cameraseventeen', \
                    'cameraeighteen', 'cameranineteen', 'cameratwenty', 'cameratwentyone', 'cameratwentytwo', \
                    'cameratwentythree', 'cameratwentyfour', 'cameratwentyfive', 'cameratwentysix', 'cameratwentyseven', \
                    'cameratwentyeight', 'cameratwentynine', 'camerathirty', 'camerathirtyone', 'camerathirtytwo', 'camerathirtythree', \
                    'camerathirtyfour', 'camerathirtyfive', 'camerathirtysix', 'camerathirtyseven', 'camerathirtyeight', 'camerathirtynine', \
                    'cameraforty', 'camerafortsyone', 'camerafortytwo', 'camerafortythree', 'camerafortyfour', 'camerafortyfive', \
                    'camerafortysix', 'camerafortyseven', 'camerafortyeight', 'camerafortynine', 'camerafifty']

    
    data_object = SplitCUHKPEDES(dataset_path="/media/rockson/Data_drive/datasets/CUHK-PEDES (backup)", category_names=category_names, seed=202405)
    global_list = data_object.split_and_copy_images()
    data_object.save_global_list_to_file(global_list)
    data_object.print_statistics()
