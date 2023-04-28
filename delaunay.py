from scipy.spatial import Delaunay
import cv2
import numpy as np

NUM_FEATURE_POINTS = 20 # a full face should have 20 feature points

def get_facial_feature_points(face):
    '''
    Finds feature points in the input face image.
    Input:
        - face: a black and white image of a face
    Output:
        - feature_points: a 2D numpy array in which each row is an (x,y) coordinate pair representing a feature point
    '''
    x, y, w, h = 0, 0, face.shape[1], face.shape[0]
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

def detect_faces(image):
    '''
    Detects faces in a black and white image.
    Input:
        - image: a black and white image.
    Output:
        - faces: a list of tuples (x, y, w, h) indicating top-left corner, and width and height of face
    '''
    face_cascade = cv2.CascadeClassifier()

    # use an opencv pretrained model for the face classifier
    face_cascade.load(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

    # detect faces
    return face_cascade.detectMultiScale(image)

def crop_faces(images, faces):
    '''
    Crops faces in a list of black and white images.
    Input:
        - image: a list of black and white images.
        - faces: a list of tuples (i, x, y, w, h) indicating index, top-left corner, and width and height of face
    Output:
        - face_images: a list of images cropped to the faces, resized to the smallest face
    '''
    smallest_w = min([face[3] for face in faces])
    smallest_h = min([face[4] for face in faces])
    all_images = []
    for (i, x, y, w, h) in faces:
        all_images.append(cv2.resize(images[i][y : y + h, x : x + w], (smallest_w, smallest_h)))
    return all_images

def get_face_keypoints(face_images):
    '''
    Gets keypoints of faces in a black and white image.
    Input:
        - face_images: a list of images cropped to the faces
    Output:
        - feature_point_list: a list of lists of 2D points, where each 2D point corresponds to a keypoint in the face
    '''
    all_feature_points = []
    for image in face_images:
        # create a triangulation for each face found in the image
        feature_points = get_facial_feature_points(image)
        all_feature_points.append(feature_points if len(feature_points) == NUM_FEATURE_POINTS else None)
    return all_feature_points

def create_triangulations(feature_point_list):
    '''
    Creates a Delaunay triangulation for each face found in the input image.
    Input:
        - feature_point_list: a list of lists of 2D points, where each 2D point corresponds to a keypoint in the face
    Output:
        - triangulations: a list of simplices
    '''
    triangulations = []
    # create a triangulation for each face found in the image
    for feature_points in feature_point_list:
        # only add triangulations for full faces
        triangulations.append(Delaunay(feature_points).simplices if feature_points is not None else None)

    return triangulations

def interpolate_triangulations(feature_points_1, feature_points_2, num_interpolations=10):
    '''
    Creates a list of interpolated triangulations of the input list of triangulations, as a function of time.
    Input:
        - feature_points_1: a list of 2D points, where each 2D point corresponds to a keypoint in the 1st face
        - feature_points_2: a list of 2D points, where each 2D point corresponds to a keypoint in the 2nd face
        - num_interpolations: how many units of time to interpolate for
    Output:
        - interpolations: a tuple of 2 lists of tuples containing feature points and simplex, one list for every unit of time
    '''
    assert feature_points_1 is not None
    assert feature_points_2 is not None

    interpolations_1, interpolations_2 = [], []
    np_points_1 = np.array(feature_points_1)
    np_points_2 = np.array(feature_points_2)
    for i in range(num_interpolations + 1):
        # Take a step function towards the other keypoints, and create the triangulation
        interpolated_1 = np_points_1 + (np_points_2 - np_points_1) * (i / num_interpolations)
        interpolated_2 = np_points_2 + (np_points_1 - np_points_2) * (i / num_interpolations)
        interpolations_1.append((interpolated_1, Delaunay(interpolated_1).simplices))
        interpolations_2.append((interpolated_2, Delaunay(interpolated_2).simplices))
    return interpolations_1, interpolations_2

