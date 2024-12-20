# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from PIL import Image, ImageFile
import errno
import json
import pickle as pkl
import os
import os.path as osp
import nltk
from nltk.data import find

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(img_path:str=None, mode:str='RGB')->Image:
    """
    Doc.:   Keep reading image until succeed. This can avoid IOError incurred by heavy IO process.
            Can read images in both RGB and greyscale depending on the mode parameter ('RGB' or 'L').
    
    Args.:  • img_path: absolute path of the image to read
            • mode: read as RGB or greyscale image. Values: `RGB` or `L`.
    
    Return: image as Image data type 
    """
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert(mode)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def get_text_embedding(path, length):
    with open(path, 'rb') as f:
        word_frequency = pkl.load(f)


def get_id_list_from_MHPv2_txt_file(text_file_path:str=None)->list:
    """
    Doc.:   Retrieves all IDs from the specified text file.

    Args.:  • text_file_path (str): The path to the text file from which IDs are to be read.

    Returns: A list of IDs found in the text file.
    """
    # Validate the text file path
    if text_file_path is None:
        raise ValueError("No text file path provided.")
    if not os.path.exists(text_file_path):
        raise ValueError(f"The file at {text_file_path} does not exist.")

    # Read IDs from the file
    with open(text_file_path, 'r') as file:
        ids = [line.strip() for line in file]

    if not ids:
        raise ValueError(f"The file at {text_file_path} contains no IDs.")

    return ids



def download_punkt_if_not_exists():
    try:
        find('tokenizers/punkt')
        print("'punkt' is already downloaded.")
    except LookupError:
        print("Downloading 'punkt'...")
        nltk.download('punkt')
        print("'punkt' downloaded successfully.")



def write_to_text_file(filename: str, data: str, write_operation: str = 'append') -> None:
    """
    Write data to a text file.

    Args:
        filename (str): The name of the file to write to.
        data (str): The data to write to the file.
        write_operation (str, optional): The write operation to perform. Defaults to 'append'.

    Raises:
        ValueError: If filename is None or empty.
        NotImplementedError: If write_operation is not supported.
    """
    if not filename:
        raise ValueError("Filename cannot be None or empty.")

    if write_operation == 'append':
        mode = 'a'
    elif write_operation == 'write':
        mode = 'w'
    else:
        raise NotImplementedError(f"Write operation '{write_operation}' is not supported.")

    with open(filename, mode) as file:
        file.write(data)
        file.write("\n")


