import dlib
import numpy as np
import cv2

# Only make these objects once (upon load)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
num_landmarks = 68

def crop_to_bounds(image, rect, required_pts=[]): 
    '''
    Input: 
        image: the image to be cropped
        rect: the rectangle representing the region that must be visible after the crop 
        required_pts: points either inside or outside the rectangle that you would 
            also like visible after the cropping
    Output: 
        The image cropped according to specifications
    '''

    # Extract bounds from rectangle
    x, y = rect.left(), rect.top()
    w, h = rect.right() - x, rect.bottom() - y 

    # Make sure rectangle doesn't overflow to end
    x = max(x, 0)
    y = max(y, 0)

    # Adjust to include required points
    for [xi, yi] in required_pts: 
        if xi < x: 
            w += abs(x - xi)
            x = xi
        if yi < y: 
            h += abs(y - yi)
            y = yi
        if xi >= x+w: 
            w = abs(xi - x)
        if yi >= y+h: 
            h = abs(yi - y)

    # Crop image
    return image[y:y+h, x:x+w]

def get_face_bounds(image): 
    '''
    Input: 
        image: an image that contains a face 
    Output: 
        the rectangle object representing the bounding box that holds the face
    '''
    global detector
    return detector(image, 1)[0]

def get_facial_landmarks(image, rect): 
    '''
    Input: 
        image: an image that contains a face 
        rect: a rectangle object that roughly contains the face
    Output: 
        A [68, 2] shape array such that each i^th element represents the x, y point 
            for the i^th facial landmark according to dlib's 68-facial landmark model
    '''
    global predictor, num_landmarks
    
    # Get keypoints
    landmarks = predictor(image, rect)
    keypoints = np.zeros((num_landmarks, 2))

    # Translate them into more usable format
    for x in range(num_landmarks): 
        keypoints[x] = [landmarks.part(x).x, landmarks.part(x).y ]

    return keypoints

def detect_face(image_name): 
    '''
    Input:
        image_name: the path to the face image (relative or absolute)
    Output: 
        0: facial landmark keypoint array in shape [68, 2]
        1: the image cropped such that the face is the correct size
        2: a bounds object that if you expand and pass into recrop, you can 
            perform the same cropping/resizing that led to the cropped face
    '''
    
    # Determine rought initial binding box
    gray_image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2GRAY)
    bounds = get_face_bounds(gray_image)
    init_keypoints = get_facial_landmarks(gray_image, bounds)

    # Make sure all face points are within the box
    required_pts = [init_keypoints.min(axis=0).astype('int64'), init_keypoints.max(axis=0).astype('int64')]
    cropped_face = crop_to_bounds(gray_image, bounds, required_pts)

    # Resize image and get facial landmarks
    cropped_face = cv2.resize(cropped_face, (500, 500))
    new_bounds = get_face_bounds(cropped_face)
    keypoints_cropped = get_facial_landmarks(cropped_face, new_bounds)

    # Make sure all facial landmarks fall within the image and add one to each of 
    # the corners so the background also shifts and morphs
    keypoints_cropped = np.clip(keypoints_cropped, 0, 499)
    keypoints_cropped = np.append(keypoints_cropped, [[0, 0], [0, 499], [499, 499], [499, 0]], axis=0)

    return keypoints_cropped, cropped_face, (bounds, required_pts) 

def recrop(image, bounds, required_pts): 
    '''
    Input: 
        image: the image to be cropped
        rect: the rectangle representing the region that must be visible after the crop 
        required_pts: points either inside or outside the rectangle that you would 
            also like visible after the cropping
    Output: 
        The image cropped and resized to match others
    '''
    cropped_face = crop_to_bounds(image, bounds, required_pts)
    cropped_face = cv2.resize(cropped_face, (500, 500))

    return cropped_face