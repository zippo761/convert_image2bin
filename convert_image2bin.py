import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy import ndimage
from pdf2image import convert_from_path


def spoil_page_pdf(path_of_file: str, number_of_page: int):
    """
    This function convert pdf to jpg and split all pages. Save page in folder.

    Parameters
    ----------
    path_of_file : str: full path of file,
    number_of_page : int.

    Returns
    -------
    None.

    """
    path = path_of_file
    pages = convert_from_path(path, 400)
    num_page = 1
    for page in pages:
        if num_page == number_of_page:
            image_name = "Page_" + str(num_page) + ".jpg"
            page.save(image_name, "JPEG")
            return image_name
        num_page += 1


def rotation(path_jpg):
    """
    This function rotating image.

    Parameters
    ----------
    path_jpg : full path of jpg file.

    Returns
    -------
    img_rotated : np.ndarray.

    """
    image = cv2.imread(path_jpg)
    img_before = image
    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100,
                            minLineLength=100, maxLineGap=5)
    angles = []
    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    median_angle = np.median(angles)
    image_rotated = ndimage.rotate(img_before, median_angle)
    # print(f"Angle is {median_angle:.04f}")
    return image_rotated


def convert_img2bin(image):
    """
    This function increase quality of image.

    Parameters
    ----------
    image : np.ndarray.

    Returns
    -------
    img : np.ndarray.

    """
    # Gaussian blur
    image = cv2.GaussianBlur(image, (1, 1), 0)
    # grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # threshold
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 30)

    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel)

    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel)

    return image


def save_jpg(path_of_file, image):
    """
    This funcion save np.ndarray  in format jpg.

    Parameters
    ----------
    path_of_file : full path of file.
    image : np.ndarray.

    Returns
    -------
    None.

    """
    cv2.imwrite(path_of_file, image)


if __name__ == '__main__':
    import time

    def timer(func):
        def wrapper(*args, **kwargs):
            before = time.monotonic()
            retval = func(*args, **kwargs)
            after = time.monotonic() - before
            print("Function {}: {} seconds".format(func.__name__, after))
            return retval
        return wrapper

    # path of pdf file
    pdf_path = r'C:/Anaconda3/envs/py33/КС2-1.pdf'

    # take all pages from pdf
    decorated_spoil_page_pdf = timer(spoil_page_pdf)(pdf_path, 3)

    # path of jpg file
    path_jpg = r'C:/Anaconda3/envs/py33/Page_3.jpg'

    # make rotation
    output_img = rotation(path_jpg)

    # boost quality of image
    output_img = convert_img2bin(output_img)

    # save file in folder
    cv2.imwrite(r'C:/Anaconda3/envs/py33/convert_img2bin.jpg', output_img)

    # show example
    plt.imshow(output_img)
    plt.show()
