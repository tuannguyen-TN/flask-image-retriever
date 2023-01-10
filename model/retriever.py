import logging as log
import sys
from time import perf_counter
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_retrieval import ImageRetrieval
from common import central_crop

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)


def time_elapsed(func, *args):
    """ Auxiliary function that helps measure elapsed time. """

    start_time = perf_counter()
    res = func(*args)
    elapsed = perf_counter() - start_time
    return elapsed, res


def Retriever(model_path, target_image, image_list, top_k):
    INPUT_SIZE = 224
    DEVICE_NAME = 'CPU'

    img_retrieval = ImageRetrieval(model_path, DEVICE_NAME, image_list, INPUT_SIZE)

    image = cv2.imread(target_image)
    # image = cv2.cvtColor(cv2.imread(filename=sys.argv[1]), code=cv2.COLOR_RGB2BGR)
    # image = cv2.imread(filename='/Users/shinobi07/Desktop/school/year4/code/dataset/lion.jpeg')

    elapsed_time = 0
    sorted_indexes = []
    start_time = perf_counter()

    if image is not None:
        image = central_crop(image, divide_by=5, shift=1)

        elapsed, probe_embedding = time_elapsed(img_retrieval.compute_embedding, image)
        elapsed_time += elapsed

        elapsed, (sorted_indexes, distances) = time_elapsed(img_retrieval.search_in_gallery, probe_embedding)
        elapsed_time += elapsed

        sorted_classes = [img_retrieval.gallery_classes[i] for i in sorted_indexes]

    result_info = [(os.path.basename(img_retrieval.impaths[i]), distances[i]) for i in sorted_indexes]
    print(f'Total elapsed time: {elapsed_time}')

    return result_info[:min(top_k, len(result_info))], min(top_k, len(result_info))

if __name__ == "__main__":
    print(Retriever(os.path.abspath('../image-retrieval-0001/FP32/image-retrieval-0001.xml'), sys.argv[1], sys.argv[2], int(sys.argv[3])))
