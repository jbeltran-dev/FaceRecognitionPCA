from PrincipalComponentAnalysis import PrincipalComponentAnalysis
from ImageMatrixConverter import ImageMatrixConverter
from Dataset import Dataset


def calculate_accuracy(correct_values, incorrect_values):
    """
    Calculate the accuracy percentage found.
    :param correct_values: Correct values.
    :param incorrect_values: Incorrect values.
    :return: Accuracy percentage found.
    """
    if correct_values + incorrect_values == 0:
        return 0.0
    else:
        accuracy_percent = correct_values / (correct_values + incorrect_values) * 100
        return accuracy_percent


# Images numbers to train process, dataset has 10 values, we need 8 to train and 2 for test
images_number_for_training = 9

# Width and height image
img_width = 50
img_height = 50

# Create an instance of the dataset class to separate images into training and testing groups
dataset = Dataset(images_number_for_training)

# Images to training process
image_paths_for_training = dataset.train_images
labels_for_training = dataset.train_labels

# Images to testing process
images_path_for_test = dataset.test_images
labels_for_test = dataset.test_labels

# Get all classes from dataset
classes = dataset.person_names

# Get image matrix from training images
image_to_matriz = ImageMatrixConverter(image_paths_for_training, img_width, img_height)
img_matrix = image_to_matriz.get_matrix()

# Principal Component Analysis process
pca = PrincipalComponentAnalysis(img_matrix, labels_for_training, classes, images_number_for_training, img_width,
                                 img_height, variance_retained_percent=90)

new_coordinates = pca.compute_pca()

# Recognizer process
correct = 0
incorrect = 0
i = 0

for image_path in images_path_for_test:

    # Get the name of the person being compared, for each image for test we have a label for test
    class_index = labels_for_test[i]
    name = classes[class_index]

    print("Face loaded: " + name)

    # Load image from path
    image = image_to_matriz.load_image_from_path(image_path)

    # Get new coordinates from image before have gotten its mean and projecting on new basis
    new_coordinates_for_image = pca.preprocess_single_image(image)

    # Get class (person name) found
    found_name = pca.recognize_faces(new_coordinates_for_image)

    print("Person recognized: " + found_name)
    print("")

    # If the name matches what PCA found, it is marked as correct
    if found_name == name:
        correct += 1
    else:
        incorrect += 1

    i = i + 1

accuracy = calculate_accuracy(correct, incorrect)

print("")
print("Correct recognitions: ", correct)
print("Incorrect recognitions: ", incorrect)
print(f"Accuracy: {accuracy:.2f}%")