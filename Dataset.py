import os


# Class to read the images dataset
class Dataset:

    """
    Constructor, it needs number of images required per person for training model
    """
    def __init__(self, required_images_per_person):

        # Dataset path
        self.dataset_directory = "Images"

        # Variables for training
        self.train_images = []
        self.train_labels = []

        # Variables for testing
        self.test_images = []
        self.test_labels = []

        # To store each person's name
        self.person_names = []

        self.load_dataset(required_images_per_person)

    """
    Load images from dataset, it needs images per person, if we have 10 images, we can to use 8 images for
    training process and use 2 images for test
    """
    def load_dataset(self, required_images_per_person):
        person_index = 0

        for person_name in os.listdir(self.dataset_directory):
            person_directory = os.path.join(self.dataset_directory, person_name)

            # Validate if is directory
            if os.path.isdir(person_directory):

                # Valid that the total number of images to train in the directory exits
                if self.has_enough_images(person_directory, required_images_per_person):

                    # Separates images into images for training and testing
                    self.process_person(person_directory, required_images_per_person, person_index)

                    # To move the next person
                    person_index += 1

    """
    Validate if directory images has number images required for training, true if is valid
    """
    @staticmethod
    def has_enough_images(directory, required_images_per_person):
        return len(os.listdir(directory)) >= required_images_per_person

    """
    Separate images into images to train and images to test
    """
    def process_person(self, person_directory, required_images_per_person, person_index):
        image_files = os.listdir(person_directory)
        images_added = 0

        # Iterate all images of the person in the selected directory
        for image_name in image_files:
            image_path = os.path.join(person_directory, image_name)

            # Util the validation of images to train is completed
            if images_added < required_images_per_person:

                # Add image to train and its label
                self.train_images.append(image_path)
                self.train_labels.append(person_index)

                # Add the name of the person being created only once
                if images_added == 0:
                    self.person_names.append(os.path.basename(person_directory))

            else:
                self.test_images.append(image_path)
                self.test_labels.append(person_index)

            images_added += 1
