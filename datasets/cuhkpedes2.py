"""
Doc.:   Codebase adapted from https://github.com/anosorae/IRRA/tree/main 
"""

# System modules
import os.path as op
from typing import List

# 3rd party modules

# Application modules
from utils.iotools import read_json

class CUHKPEDES(object):
   
    def __init__(self, config:dict):
        """
        Doc.:   CUHK-PEDES dataset from Person Search with Natural Language Description (https://arxiv.org/pdf/1702.05729.pdf)

                Dataset Statistics
                -------------------
                • train split:  34,054 images and 68,126 descriptions for 11,003 persons (ID: 1-11003)
                • val split:    3,078  images and 6,158 descriptions for 1,000 persons (ID: 11004-12003)
                • test split:   3,074  images and 6,156 descriptions for 1,000 persons (ID: 12004-13003)

                Totals:
                -------------------
                • images: 40,206
                • persons: 13,003
                
                annotation format: 
                [{'split', str,
                'captions', list,
                'file_path', str,
                'processed_tokens', list,
                'id', int}...]

                Because we will use the IDs as class labels, so we will have to start from 0. 
                So instead of 1~11003, we will do 0~11002. Therefore the splits will be
                • train (0-11002)
                • val (0-999)
                • test (0-999)
        """
        super(CUHKPEDES, self).__init__()
        self.dataset_dir = config.dataset_path
        self.img_dir = op.join(self.dataset_dir, 'imgs/')
        self.anno_path = op.join(self.dataset_dir, 'reid_raw.json')
        
        # Lets use these offset points to make the IDs start from 0
        # So depending on the split type, we shall do ID-self.ID_offset_point[split_type]
        # This way we will ensure that all human labels start from 0
        self.ID_offset_point = dict(train=1, # (ID: 1-11003)
                                      val=11004, # (ID: 11004-12003)
                                      test=12004) # (ID: 12004-13003)

        self._check_before_run()
        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)
        self.train, self.train_id_container = self._process_anno(self.train_annos,'train') 
        self.val, self.val_id_container = self._process_anno(self.val_annos,'val') 
        self.test, self.test_id_container = self._process_anno(self.test_annos,'test')
        
        

    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos
    

    def _process_anno(self, annos: List[dict],split=None):
        if split == None:
            raise ValueError("`split` value can not be none`")
        print("The dataset split is ",split)
        pid_container = set() # labels
        dataset = []
        image_id = 0
        for anno in annos:
            #pid = int(anno['id']) - 1 # make pid begin from 0
            pid = int(anno['id']) - int(self.ID_offset_point[split])

            pid_container.add(pid)
            img_path = op.join(self.img_dir, anno['file_path'])
            captions = anno['captions'] # caption list
            for caption in captions:
                dataset.append((pid, image_id, img_path, caption))
            image_id += 1

        pid_container_list = list(pid_container)
        pid_container_list.sort()

        # check pid begin from 0 and no break
        for idx, pid in enumerate(pid_container_list):
            assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
    
        return dataset, pid_container

        

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
        