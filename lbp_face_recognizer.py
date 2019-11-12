import os
import cv2
import numpy as np

subjects = ['', 'Mike Ross', 'Harvey Specter']

def detect_faces(image):
    
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_cascades = cv2.CascadeClassifier(os.path.join(os.getcwd(), 'lbpcascade/lbpcascade_frontalface.xml'))

    faces = faces_cascades.detectMultiScale(grayscale, scaleFactor=1.2, minNeighbors=5)

    if (len(faces) == 0):
        return None, None

    x, y, w, h = faces[0]

    return grayscale[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces, labels = [], []

    for dir in dirs:
        if not (dir.startswith('person')):
            continue

        label = int(dir.replace('person', ''))
        subject_dir_path = os.path.join(data_folder_path, dir)
        images = os.listdir(subject_dir_path)

        for image in images:
            if image.startswith('.'):
                continue
            image_path = os.path.join(subject_dir_path, image)
            img = cv2.imread(image_path)
            cv2.imshow('The current subject of training', img)
            cv2.waitKey(100)

            face, rect = detect_faces(img)

            if face is not None:
                faces.append(face)
                labels.append(label)

            cv2.waitKey(1)
            cv2.destroyAllWindows()

    return  faces, labels

def draw_rectangle(img, rect):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), ((x+w), (y+h)), (0, 0, 255), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)


if __name__ == '__main__':
    print('Preparing data...')
    faces, labels = prepare_training_data('data')
    print('Data prepared')

    print('Total faces:', len(faces))
    print('Total labels:', len(labels))

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_recognizer.train(faces, np.array(labels))

    test_images_path = os.path.join(os.getcwd(), 'test_data')

    test_images = os.listdir(test_images_path)
    print(test_images)

    for test_image in test_images:
        img = test_image
        img_mat = cv2.imread(os.path.join(test_images_path, img))

        face, rect = detect_faces(img_mat)

        label = face_recognizer.predict(face)
        label_name = subjects[label[0]]

        draw_rectangle(img_mat, rect)
        draw_text(img_mat, label_name, rect[0], rect[1])

        cv2.imshow(label_name, img_mat)
        cv2.waitKey(0)


