import os
import cv2
from imgaug import augmenters as iaa

def augment_images(input_dir, output_dir, num_augmentations=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  
        iaa.Affine(rotate=(-10, 10)), 
        iaa.GaussianBlur(sigma=(0, 1.0)) 
    ])

    for person_name in os.listdir(input_dir):
        person_input_dir = os.path.join(input_dir, person_name)
        person_output_dir = os.path.join(output_dir, person_name)
        if not os.path.exists(person_output_dir):
            os.makedirs(person_output_dir)

        for image_file in os.listdir(person_input_dir):
            image_path = os.path.join(person_input_dir, image_file)
            image = cv2.imread(image_path)

            for i in range(num_augmentations):
                augmented_image = seq.augment_image(image)
                output_path = os.path.join(person_output_dir, f"{image_file.split('.')[0]}_aug_{i}.jpg")
                cv2.imwrite(output_path, augmented_image)


input_directory = 'test_dataset'
output_directory = 'augmented_dataset'
augment_images(input_directory, output_directory)
