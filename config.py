import secrets
import os

SAVE_FOLDER = os.path.abspath('./dataset')
SECRET_KEY = secrets.token_hex() # required by Flask to upload images
IMAGES = {}
TARGET_IMAGE = ''
IMAGE_LIST = os.path.join(SAVE_FOLDER, 'images.txt')
PREDICTIONS = []
MODEL_PATH = os.path.abspath('image-retrieval-0001/FP32/image-retrieval-0001.xml')
TOP_K = 0