import cv2
import numpy as np


class ImageMatrixConverter:
    """
    A class to convert a list of images into a matrix representation,
    where each image is a column vector in the matrix.
    """

    def __init__(self, image_paths=None, image_width=None, image_height=None):
        """
        Constructor for ImageMatrixConverter.

        :param image_paths: List of paths to images.
        :param image_width: Width to resize images.
        :param image_height: Height to resize images.
        """

        # Return an empty path
        if image_paths is None:
            image_paths = []

        self.image_paths = image_paths
        self.image_width = image_width
        self.image_height = image_height

    def get_matrix(self):
        """
        Get an image matrix (each image will be a column matrix)
        :return: Matrix with all images as a column
        """

        # Because each image must be added as a column in a matrix
        columns = len(self.image_paths)

        # To get the total number of the matrix rows
        image_size = self.image_width * self.image_height

        # Create an empty matrix with zeros, to add images later
        images_matrix = np.zeros((image_size, columns))

        # Read each image from dataset
        for i, path in enumerate(self.image_paths):

            # Load image resized in grayscale
            gray_image_resized = self.load_image_from_path(path)
 
            # Convert image in grayscale to a vector, because we need to add it as a column in a new matrix
            image_vector = self.image_to_vector(gray_image_resized)

            # Add image gray vector in a matrix with zeros (each image is one column)
            images_matrix[:, i] = image_vector

        return images_matrix

    def load_image_from_path(self, path):
        """
        Load an image from file path, converts it to grayscale and resize it.

        :param path: File path to the image.
        :return: Resized grayscale image.
        """

        # Load image in grayscale
        gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # Resize the grayscale image
        resized_image = cv2.resize(gray_image, (self.image_width, self.image_height))

        return resized_image

    @staticmethod
    def image_to_vector(image):
        """
        Convert a grayscale image to a vector representation.

        :param image: Grayscale image.
        :return: Vector representation of the image.
        """
        return image.reshape(-1)

    @staticmethod
    def show_images(label_to_show, image_matrix):
        """
        Show an image with a label.
        :param label_to_show: Label to show.
        :param image_matrix: Matrix representation of the image.
        :return: Showing the image.
        """

        cv2.imshow(label_to_show, image_matrix)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
