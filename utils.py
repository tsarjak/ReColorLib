import numpy as np
import cv2
from PIL import Image


class Transforms:
    """
    Holds transformation matrices.
    All the details for transformation matrices can be found at: https://arxiv.org/pdf/1711.10662.pdf.
    """

    @staticmethod
    def rgb_to_lms():
        return np.array([[17.8824, 43.5161, 4.11935],
                         [3.45565, 27.1554, 3.86714],
                         [0.0299566, 0.184309, 1.46709]]).T

    @staticmethod
    def lms_to_rgb():
        return np.array([[0.0809, -0.1305, 0.1167],
                         [-0.0102, 0.0540, -0.1136],
                         [-0.0004, -0.0041, 0.6935]]).T

    @staticmethod
    def lms_protanopia(degree: float = 1.0):
        return np.array([[1 - degree, 2.02344 * degree, -2.52581 * degree],
                         [0, 1, 0],
                         [0, 0, 1]]).T

    @staticmethod
    def lms_deutranopia(degree: float = 1.0):
        return np.array([[1, 0, 0],
                         [0.494207 * degree, 1 - degree, 1.24827 * degree],
                         [0, 0, 1]]).T

    @staticmethod
    def lms_tritanopia(degree: float = 1.0):
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [-0.395913 * degree, 0.801109 * degree, 1 - degree]]).T

    @staticmethod
    def hybrid_protanomaly_deuteranomaly(degree_p: float = 1.0, degree_d: float = 1.0):
        return np.array([[1 - degree_p, 2.02344 * degree_p, -2.52581 * degree_p],
                         [0.494207 * degree_d, 1 - degree_d, 1.24827 * degree_d],
                         [0, 0, 1]]).T

    @staticmethod
    def correction_matrix(protanopia_degree, deutranopia_degree):
        return np.array([[1 - deutranopia_degree/2, deutranopia_degree/2, 0],
                         [protanopia_degree/2, 1 - protanopia_degree/2, 0],
                         [protanopia_degree/4, deutranopia_degree/4, 1 - (protanopia_degree + deutranopia_degree)/4]]).T


class Utils:

    @staticmethod
    def load_rgb(path):
        img_rgb = np.array(Image.open(path)) / 255
        return img_rgb

    @staticmethod
    def load_lms(path):
        img_rgb = np.array(Image.open(path)) / 255
        img_lms = np.dot(img_rgb[:,:,:3], Transforms.rgb_to_lms())

        return img_lms
