import os
import cv2

def preprocess_images(input_dir, output_dir, target_size=(480, 480)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for person_name in os.listdir(input_dir):
        person_input_dir = os.path.join(input_dir, person_name)
        person_output_dir = os.path.join(output_dir, person_name)
        if not os.path.exists(person_output_dir):
            os.makedirs(person_output_dir)

        for image_file in os.listdir(person_input_dir):
            image_path = os.path.join(person_input_dir, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                # Resize image
                resized_image = cv2.resize(image, target_size)

                output_path = os.path.join(person_output_dir, image_file)
                cv2.imwrite(output_path, resized_image)

# Example usage
input_directory = 'augmented_dataset'
output_directory = 'preprocessed_dataset'
preprocess_images(input_directory, output_directory, target_size=(640, 640))
