import numpy as np
from PIL import Image
import cv2

from utils import Transforms, Utils


class Core:

    @staticmethod
    def simulate(input_path: str,
                 simulate_type: str = 'protanopia',
                 simulate_degree_primary: float = 1.0,
                 simulate_degree_sec: float = 1.0,
                 return_type: str = 'save',
                 save_path: str = None):
        """
        :param input_path: Input path of the image.
        :param simulate_type: Type of simulation needed. Can be 'protanopia', 'deutranopia', 'tritanopia', 'hybrid'.
        :param simulate_degree_primary: Primary degree of simulation: used for 'protanopia', 'deutranopia', 'tritanopia'
        :param simulate_degree_sec: Secondnary degree of simulation: used for 'hybrid'.
        :param return_type: How to return the Simulated Image. Use 'pil' for PIL.Image, 'np' for Numpy array,
                            'save' for Saving to path.
        :param save_path: Where to save the simulated file. Valid only if return_type is set as 'save'.
        :return:
        """

        assert simulate_type in ['protanopia', 'deutranopia', 'tritanopia', 'hybrid'], \
            'Invalid Simulate Type: {}'.format(simulate_type)

        # Load the image file in LMS colorspace
        img_lms = Utils.load_lms(input_path)

        if simulate_type == 'protanopia':
            transform = Transforms.lms_protanopia(degree=simulate_degree_primary)
        elif simulate_type == 'deutranopia':
            transform = Transforms.lms_deutranopia(degree=simulate_degree_primary)
        elif simulate_type == 'tritanopia':
            transform = Transforms.lms_tritanopia(degree=simulate_degree_primary)
        else:
            transform = Transforms.hybrid_protanomaly_deuteranomaly(degree_p=simulate_degree_primary,
                                                                    degree_d=simulate_degree_sec)

        # Transforming the LMS Image
        img_sim = np.dot(img_lms, transform)

        # Converting back to RGB colorspace
        img_sim = np.uint8(np.dot(img_sim, Transforms.lms_to_rgb()) * 255)

        if return_type == 'save':
            assert save_path is not None, 'No save path provided.'
            cv2.imwrite(save_path, img_sim)
            return

        if return_type == 'np':
            return img_sim

        if return_type == 'pil':
            return Image.fromarray(img_sim)

    @staticmethod
    def correct(input_path: str,
                protanopia_degree: float = 1.0,
                deutranopia_degree: float = 1.0,
                return_type: str = 'save',
                save_path: str = None
                ):
        """
        Use this method to correct images for People with Colorblindness. The images can be corrected for anyone
        having either protanopia, deutranopia, or both. Pass protanopia_degree and deutranopia_degree as diagnosed
        by a doctor using Ishihara test.
        :param input_path: Input path of the image.
        :param protanopia_degree: Protanopia degree as diagnosed by doctor using Ishihara test.
        :param deutranopia_degree: Deutranopia degree as diagnosed by doctor using Ishihara test.
        :param return_type: How to return the Simulated Image. Use 'pil' for PIL.Image, 'np' for Numpy array,
                            'save' for Saving to path.
        :param save_path: Where to save the simulated file. Valid only if return_type is set as 'save'.
        :return:
        """

        img_rgb = Utils.load_rgb(input_path)

        transform = Transforms.correction_matrix(protanopia_degree=protanopia_degree,
                                                 deutranopia_degree=deutranopia_degree)

        img_corrected = np.uint8(np.dot(img_rgb, transform) * 255)

        if return_type == 'save':
            assert save_path is not None,'No save path provided.'
            cv2.imwrite(save_path, img_corrected)
            return

        if return_type == 'np':
            return img_corrected

        if return_type == 'pil':
            return Image.fromarray(img_corrected)
