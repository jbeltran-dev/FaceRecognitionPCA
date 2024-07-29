import numpy as np
import scipy.linalg as la


class PrincipalComponentAnalysis:
    """
    Class to implement the Principal Component Analysis algorithm for images dataset
    Constructor, it needs all information.
    """

    def __init__(self
                 , images_matrix
                 , images_labels
                 , person_names
                 , required_images_number
                 , images_width
                 , images_height
                 , variance_retained_percent):
        """
        :param images_matrix: Image matrix, each image represents a column
        :param images_labels: Image tag, or image name (ex. 1.jpg, something.jpg ...)
        :param person_names: Classes in the dataset, in this case, represents a person's name
        :param required_images_number: The number of images required for training
        :param images_width: Width of each image
        :param images_height: Height of each image
        :param variance_retained_percent: Desired percentage of variance
        """

        self.images_matrix = images_matrix
        self.images_labels = images_labels
        self.person_names = person_names
        self.required_images_number = required_images_number
        self.images_width = images_width
        self.images_height = images_height
        self.variance_retained_percent = variance_retained_percent
        self.mean_face = None
        self.new_bases = None
        self.new_coordinates = None

        # Subtracts the mean face from the image matrix
        self.subtract_mean_face()

    def subtract_mean_face(self):
        """
        Subtracts the mean face. Gets M = R - μ, where [R] it's image matrix and [μ] it's de mean
        """
        # We need to get the mean image, in other words its average values for each row, because its parameter 1
        mean_image = np.mean(self.images_matrix, axis=1)

        # Since main_image is a vector, we need to convert it to matrix and reshape into a column. (using the transpose)
        self.mean_face = np.asmatrix(mean_image).T

        self.images_matrix -= self.mean_face

    def calculate_p_for_variance(self, eigen_values):
        """
        Calculates the number of principal components (P) required to retain the specified percentage of variance

        :param eigen_values: eigenvalues from images matrix
        :return: Return number of principal components needed to rach or exceed the specified percentage of variance
        """
        # Summing total of the eigenvalues; this represents the total variance explained by all principal components.
        total_variance = np.sum(eigen_values)

        # Calculate the percentage of variance to be retained
        sum_threshold = total_variance * self.variance_retained_percent / 100

        # This variable will accumulate the variance as we iterate through the eigenvalues
        cumulative_variance = 0

        # To count the number of principal components summed until reaching or exceeding the variance threshold
        p = 0

        while cumulative_variance < sum_threshold:
            cumulative_variance += eigen_values[p]

            # We increment P by 1 to move the next eigenvalue
            p += 1

        # Principal components
        return p

    def compute_pca(self):
        """
        Computes Principal Component Analysis (PCA) on the image matrix.

        Perform Singular Value Decomposition (SVD) to find the eigenvectors of the covariance matrix
        u: Matrix of left singular vectors
        eigen_values: Singular values
        vt: Transpose of the matrix of right singular vectors (not required)
        """
        [u, eigen_values, _] = la.svd(self.images_matrix, full_matrices=True)

        # Convert U matrix into matrix numpy
        u_matrix = np.matrix(u)

        # Calculate number of principal components (P) to retain desired variance
        p = self.calculate_p_for_variance(eigen_values)

        # Select the first P principal components from U matrix.
        self.new_bases = u_matrix[:, 0:p]

        # Project the original images onto the new basis
        self.new_coordinates = np.dot(self.new_bases.T, self.images_matrix)

        # Return transformed coordinates of the original images in the space of principal components
        return self.new_coordinates

    def preprocess_single_image(self, single_image):
        """
        Preprocesses a single image by subtracting the mean face and projecting it onto a new basis.

        :param single_image: Single image to preprocess.
        :return: New coordinates of the preprocessed image in the space of principal components.
        """

        # Because we need the image as a column
        image_vector = np.asmatrix(single_image).ravel().T

        # Calculate a new mean to include the individual image
        new_mean = ((self.mean_face * len(self.images_labels)) + image_vector) / (len(self.images_labels) + 1)

        # Subtract the calculated new mean from the image vector
        image_vector = image_vector - new_mean

        # Project the preprocessed image onto a new basis
        preprocessed_image_coordinates = np.dot(self.new_bases.T, image_vector)

        return preprocessed_image_coordinates

    def recognize_faces(self, new_coordinates_of_image):
        """
        Recognizes the face represented by new coordinates using nearest neighbor classification.

        :param new_coordinates_of_image: Represents the new coordinates of the image to be recognized.
        :return: Name of the recognized person
        """

        # Number of classes in dataset (persons)
        total_classes = len(self.person_names)

        # List to store distances from new image coordinates to class means
        distances = []

        trainer_number_image = self.required_images_number

        # Iterate over each person/class
        for person_index in range(total_classes):

            # Extract images corresponding to the current class
            start_index = person_index * trainer_number_image
            end_index = (start_index + trainer_number_image - 1)
            images_for_person = self.new_coordinates[:, start_index:end_index]

            # Calculate the mean vector of images for the current class
            mean_image_for_person = np.mean(images_for_person, axis=1)

            # Calculate Euclidean distance between new image coordinates and mean image vector
            distance_to_mean = np.linalg.norm(new_coordinates_of_image - mean_image_for_person)

            # Store the calculated distance
            distances.append(distance_to_mean)

            print("Person : " + str(person_index) + " : Distance = " + str(distance_to_mean) + " ")

        minimum_distance = min(distances)
        maximum_distance = max(distances)
        difference = maximum_distance - minimum_distance

        minimum_distance_index = np.argmin(distances)
        recognized_person_name = self.person_names[minimum_distance_index]

        print(f"Recognized as: {recognized_person_name}")
        print(f"Difference was: {difference}")

        # Find the index of the minimum distance, which corresponds to the recognized person/class
        if difference > 910:
            person_name = recognized_person_name
        else:
            person_name = "Unknown"

        return person_name
