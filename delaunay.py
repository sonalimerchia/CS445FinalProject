from scipy.spatial import Delaunay
import cv2
import numpy as np

NUM_FEATURE_POINTS = 20 # a full face should have 20 feature points

def get_facial_feature_points(face, x, y, w, h):
    '''
    Finds feature points in the input face image.
    Input:
        - face: a black and white image of a face
        - x: the x coordinate of the top left corner of the face image relative to the larger image
        - y: the y coordinate of the top left corner of the face image relative to the larger image
        - w: the width of the face image
        - h: the height of the face image
    Output:
        - feature_points: a 2D numpy array in which each row is an (x,y) coordinate pair representing a feature point
    '''
    eyes_cascade = cv2.CascadeClassifier()
    mouth_cascade = cv2.CascadeClassifier()

    # use opencv pretrained models for the eyes and mouth classifiers
    eyes_cascade.load(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_eye.xml'))
    mouth_cascade.load(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_smile.xml'))

    feature_points = []
    
    # add facial feature points (corners and center of face bounding box)
    feature_points.append((x, y))
    feature_points.append((x + w, y))
    feature_points.append((x, y + h))
    feature_points.append((x + w, y + h))
    feature_points.append((x + w // 2, y + h // 2))

    # detect eyes
    eyes = sorted(eyes_cascade.detectMultiScale(face), key=lambda x: x[0])[:2] # [left eye, right eye]
    for (x2, y2, w2, h2) in eyes:
        # add eye feature points (corners and center of eye bounding box)
        feature_points.append((x + x2, y + y2))
        feature_points.append((x + x2 + w2, y + y2))
        feature_points.append((x + x2, y + y2 + h2))
        feature_points.append((x + x2 + w2, y + y2 + h2))
        feature_points.append((x + x2 + w2 // 2, y + y2 + h2 // 2))
    
    # detect mouth
    # the lowest mouth found is most likely to be the real mouth
    x2, y2, w2, h2 = sorted(mouth_cascade.detectMultiScale(face), key=lambda x: x[1] + x[3], reverse=True)[0]
    # add mouth feature points (corners and center of mouth bounding box)
    feature_points.append((x + x2, y + y2))
    feature_points.append((x + x2 + w2, y + y2))
    feature_points.append((x + x2, y + y2 + h2))
    feature_points.append((x + x2 + w2, y + y2 + h2))
    feature_points.append((x + x2 + w2 // 2, y + y2 + h2 // 2))

    return np.array(feature_points)

def create_triangulations(image):
    '''
    Creates a Delaunay triangulation for each face found in the input image.
    Input:
        - image: a black and white image
    Output:
        - triangulations: a list of tuples (one tuple per full face found in the image) of the form (feature_points, simplices)
    '''
    face_cascade = cv2.CascadeClassifier()

    # use an opencv pretrained model for the face classifier
    face_cascade.load(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

    # detect faces
    faces = face_cascade.detectMultiScale(image)

    triangulations = []
    
    for (x, y, w, h) in faces:
        # create a triangulation for each face found in the image
        feature_points = get_facial_feature_points(image[y : y + h, x : x + w], x, y, w, h)
        if len(feature_points) == NUM_FEATURE_POINTS:
            # only add triangulations for full faces
            triangulations.append((feature_points, Delaunay(feature_points).simplices))

    return triangulations