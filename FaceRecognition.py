from PrincipalComponentAnalysis import PrincipalComponentAnalysis
from ImageMatrixConverter import ImageMatrixConverter
from Dataset import Dataset
from datetime import datetime
import time
import cv2


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

def process_frame(frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imagen_ecualizada = cv2.equalizeHist(gray_frame)
        
        faces_detected = face_cascade.detectMultiScale(
            imagen_ecualizada, 
            scaleFactor=1.1, 
            minNeighbors=8
        )
        
        if len(faces_detected) == 0:
            faces_detected = face_cascade_profile_face.detectMultiScale(
                imagen_ecualizada,
                scaleFactor=1.1,
                minNeighbors=8
            )
            if len(faces_detected) > 0:
                print("DETECCION ROSTRO PERFIL")
       

        for (x, y, w, h) in faces_detected:
            gray_face = imagen_ecualizada[y:y + h, x:x + w]
            image = cv2.resize(gray_face, (img_height, img_width))
            record_color = (0, 0, 255)
            record_stroke = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), record_color, record_stroke)
            
            new_coordinates_for_image = pca.preprocess_single_image(image)
            name = pca.recognize_faces(new_coordinates_for_image)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_color = (255, 255, 255)
            font_stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, font_color, font_stroke, cv2.LINE_AA)
            
            print("Es: ", name)

        cv2.imshow('Colored Frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            return False
        return True

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
        # Show the image
        #ImageMatrixConverter.show_images("It's: " + name, image)
        incorrect += 1

    i = i + 1

accuracy = calculate_accuracy(correct, incorrect)

print("")
print("Correct recognitions: ", correct)
print("Incorrect recognitions: ", incorrect)
#print(f"Accuracy: {accuracy:.2f}%")

# Open video camera
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
face_cascade_profile_face = cv2.CascadeClassifier('cascades/haarcascade_profileface.xml')
url = 'https://api.verkada.com/stream/cameras/v1/footage/stream/stream.m3u8?org_id=fb775f0c-feef-4aa9-9197-5071aacaa0bf&camera_id=9fc6869c-867b-435b-ada9-4017da1e78b4&start_time=0&end_time=0&codec=hevc&resolution=low_res&type=stream&jwt=jwt_vkda_jwt_eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkYTA0YmMxYi1lYTk4LTRmNGYtOGMxZi03NjY0YmI0MTRmNWYiLCJleHAiOjE3MjE5MjU3MzEsImlhdCI6MTcyMTkyMjEzMX0.4GYa_JFq7BltThKXjc57LMHPrJPTIL43BHf8TgCEJlEIdXEDPhQMpwHNzcCQW3iCi37Qj08seId_yMJmr4Qdkg&transcode=false'
reconnect_delay = 5  # Tiempo de espera en segundos antes de intentar reconectar
while True:    
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Error: No se puede abrir la transmisión de video.")
        time.sleep(reconnect_delay)
        continue

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("Fin de la transmisión o error al leer el frame.")
            break
        
        if not process_frame(frame):
            cap.release()
            cv2.destroyAllWindows()
            exit()
        
        hora_actual = datetime.now().time()
        print("La hora actual es:", hora_actual)
    
    print("Intentando reconectar en", reconnect_delay, "segundos...")
    time.sleep(reconnect_delay)


    

