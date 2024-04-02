import cv2
import face_recognition
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

def load_face_recognition_model(model_file):
    model = load_model(model_file)
    return model

def get_class_names(class_names_file):
    class_names = {}
    with open(class_names_file, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            class_names[int(parts[1].strip())] = parts[0].strip()
    return class_names

def recognize_faces(input_type, input_path, output_dir, model, class_names_file):
    class_names = get_class_names(class_names_file)

    if input_type == '0':
        video_capture = cv2.VideoCapture(0)
    elif input_type == '1':
        video_capture = cv2.VideoCapture(input_path)
    elif input_type == '2':
        frame = cv2.imread(input_path)
        face_locations = face_recognition.face_locations(frame)
        for (top, right, bottom, left) in face_locations:
            face_image = frame[top:bottom, left:right]
            face_image = cv2.resize(face_image, (240, 240))  # Changed image size
            face_image = image.img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)
            face_image = face_image/255
            matches = model.predict(face_image)
            name = "Unknown"
            if np.max(matches) > 0.5:
                predicted_index = np.argmax(matches)
                if predicted_index in class_names:
                    name = class_names[predicted_index]
                else:
                    name = f"Person {predicted_index}"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        output_path = os.path.join(output_dir, 'output_image.jpg')
        cv2.imwrite(output_path, frame)
        cv2.imshow('Output Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        print("Invalid input type.")
        return

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(os.path.join(output_dir, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while True:
        ret, frame = video_capture.read()

        face_locations = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in face_locations:
            face_image = frame[top:bottom, left:right]
            face_image = cv2.resize(face_image, (240, 240))  # Changed image size
            face_image = image.img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)
            face_image = face_image/255
            matches = model.predict(face_image)
            name = "Unknown"

            if np.max(matches) > 0.5:
                predicted_index = np.argmax(matches)
                if predicted_index in class_names:
                    name = class_names[predicted_index]
                else:
                    name = f"Person {predicted_index}"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('Video', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

model_file = 'my_model.keras'
class_names_file = 'class_names.txt'
model = load_face_recognition_model(model_file)

input_type = input("Enter input type (webcam=0, video=1, image=2): ")
input_path = input("Enter input path: ")
output_dir = 'result'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

recognize_faces(input_type, input_path, output_dir, model, class_names_file)
