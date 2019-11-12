import os
import cv2
import numpy as np

subjects = ['', 'Mike Ross', 'Harvey Specter']  # the possible labels of the images

def detect_faces(image):
    ''' a function that takes the image as an argument, detect image and returns the region of interest(roi) '''
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # translation to grayscale 
    faces_cascades = cv2.CascadeClassifier(os.path.join(os.getcwd(), 'lbpcascade/lbpcascade_frontalface.xml')) # setting lbp as a classifier 

    faces = faces_cascades.detectMultiScale(grayscale, scaleFactor=1.2, minNeighbors=5) # detect faces

    if (len(faces) == 0): # checks if the faces list is empty or not 
        return None, None

    x, y, w, h = faces[0] # get the peremters of the first face 

    return grayscale[y:y+w, x:x+h], faces[0] 

def prepare_training_data(data_folder_path):
    ''' a function that takes the data folder path as an argument, then trains data to recognize the face and return the face labeled '''
    dirs = os.listdir(data_folder_path) # list all the sub-directories
    faces, labels = [], []

    for dir in dirs: # for each directory under argument path chosen 
        if not (dir.startswith('person')): # checking if the folder starts with 'person'
            continue

        label = int(dir.replace('person', '')) # representing the label with a fixed number
        subject_dir_path = os.path.join(data_folder_path, dir) # getting the sub-directory path 
        images = os.listdir(subject_dir_path) # setting path to image 

        for image in images: # for each image in the folder where all images exist 
            if image.startswith('.'):  # checking if he image has no label
                continue
            image_path = os.path.join(subject_dir_path, image) # intializing image path
            img = cv2.imread(image_path) # reads image 
            cv2.imshow('The current subject of training', img) # shows image
            cv2.waitKey(100)

            face, rect = detect_faces(img) #initialize the face and the rectangle from the method detect_faces

            if face is not None: # assign the face & label to the lists if the case isn't empty
                faces.append(face)
                labels.append(label)

            cv2.waitKey(1)
            cv2.destroyAllWindows()

    return  faces, labels 

def draw_rectangle(img, rect): 
    ''' function that draws the rectangle  '''
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), ((x+w), (y+h)), (0, 0, 255), 2) #

def draw_text(img, text, x, y):
        ''' function that writes the text '''
    cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)


if __name__ == '__main__':
    print('Preparing data...')
    faces, labels = prepare_training_data('data')
    print('Data prepared')

    print('Total faces:', len(faces))
    print('Total labels:', len(labels))

    face_recognizer = cv2.face.LBPHFaceRecognizer_create() # create recognizer 

    face_recognizer.train(faces, np.array(labels)) # fill recognizer with the face required to scan 

    test_images_path = os.path.join(os.getcwd(), 'test_data') # get the image that we will test from 

    test_images = os.listdir(test_images_path)
    print(test_images) # print all images he was training on 

    for test_image in test_images:
        img = test_image
        img_mat = cv2.imread(os.path.join(test_images_path, img)) # read the images we will be testing on 

        face, rect = detect_faces(img_mat) 

        label = face_recognizer.predict(face) # predict the person by the train he's been doing 
        label_name = subjects[label[0]] # put the label on the face recognized

        draw_rectangle(img_mat, rect)
        draw_text(img_mat, label_name, rect[0], rect[1]) 
        # draw on it the rectangle and the text

        cv2.imshow(label_name, img_mat)
        cv2.waitKey(0) # exit by any key 

