import cv2
import os
import numpy as np

class ImageHelper:
    @staticmethod
    def get_links(num_test: int = 1, all: bool = False, dataset: str = "dataset_old") -> tuple[list, list]:
        """
        Get links of images from the specified dataset.

        :param num_test: Number of tests.
        :param all: If True, get links of all images.
        :param dataset: Name of the dataset.
        :return: Tuple of lists containing links of x and y images.
        """
        links_x = list()
        links_y = list()
        if all: 
            for i in range(5):
                links_xy = ImageHelper.get_links(i + 1)
                links_x += links_xy[0]
                links_y += links_xy[1]
        else:      
            for directory in os.listdir(f"{dataset}/termal"):
                if os.path.isfile(f"{dataset}/termal/{directory}/{num_test}.jpg"):
                    links_x.append(f"{dataset}/termal/{directory}/{num_test}.jpg")
                    links_y.append(f"{dataset}/visible/{directory}/{num_test}.jpg")
        return links_x, links_y
    
    @staticmethod
    def resize_and_crop(image: np.ndarray, target_width: int = 10, target_height: int = 10, isheight: bool = True) -> np.ndarray:
        """
        Resize and crop the image to the target width and height.

        :param image: Input image.
        :param target_width: Target width.
        :param target_height: Target height.
        :param isheight: If True, resize based on height.
        :return: Resized and cropped image.
        """
        original_height, original_width = image.shape
        aspect_ratio = original_width / original_height

        if isheight:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image
    
    @staticmethod
    def sharpen_image(image_path: str, new_image_path: str) -> None:
        """
        Sharpen the image and save it to a new path.

        :param image_path: Path of the input image.
        :param new_image_path: Path to save the sharpened image.
        """
        image = cv2.imread(image_path)

        # Define the sharpening kernel
        sharpen_filter = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])

        sharpened_image = cv2.filter2D(image, -1, sharpen_filter)
        cv2.imwrite(new_image_path, sharpened_image)
        
    @staticmethod
    def get_pictures(links) -> np.ndarray:
        """
        Get pictures from the specified links.

        :param links: List of links.
        :return: Array of pictures.
        """
        result = list()
        for filename in links:
            imageX = cv2.imread(filename, 0).astype("float64")
            result.append(imageX)
        return np.array(result)