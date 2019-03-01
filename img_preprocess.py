import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_file_and_convert_to_binary(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    print(ret)
    return binary
