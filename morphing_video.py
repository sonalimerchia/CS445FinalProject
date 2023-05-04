
from os import listdir
from tqdm import tqdm
import random
import facial_detection
import cross_dissolve
import delaunay
import video_util
import numpy as np
import argparse
from tqdm import tqdm
import cv2

def get_image_names(directory): 
    images = []

    filenames = listdir(directory)
    random.shuffle(filenames)

    for image in filenames: 
        if image.endswith('jpeg') or image.endswith('jpg') or image.endswith('png'):
            images.append(directory + image)

    return images

def load_images(image_names): 
    failed_names = []
    
    keypoints = []
    faces = []

    print('Loading images...')
    for i in tqdm(image_names): 
        keypoints_cropped, cropped_faces, bounds_lst = facial_detection.detect_faces(i)
        if len(keypoints_cropped) == 0: 
            failed_names.append(i) 
            continue

        im = cv2.imread(i)
        for pts, bds in zip(keypoints_cropped, bounds_lst): 
            keypoints.append(pts)
            faces.append(facial_detection.recrop(im, *bds))
        
    if len(failed_names) > 0: 
        print("Failed to read/find faces in:")
        for f in failed_names: 
            print(f)

    return keypoints, faces

def make_transitions(keypoints, faces): 
    num_interpolations = 30 
    all_transitions = []
    num_faces = len(keypoints)

    print('Making morph images...')
    for face1_idx in tqdm(range(num_faces - 1)):
        keypoints1 = keypoints[face1_idx]
        keypoints2 = keypoints[face1_idx + 1]
        interpolation = delaunay.interpolate_triangulations(keypoints1, keypoints2, num_interpolations=num_interpolations)

        face1 = faces[face1_idx]
        face2 = faces[face1_idx + 1]
        image_progression = cross_dissolve.dissolve(*interpolation, [face1, face2])

        for im in image_progression: 
            all_transitions.append(im)

    return np.array(all_transitions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MorphingVideo",
        description="Makes a video of faces morphing from folder of images"
    )

    parser.add_argument('-d', '--directory', required=True)
    parser.add_argument('-f', '--output', required=True)
    args = parser.parse_args()

    if not args.output.lower().endswith('.mp4'): 
        print("Output file must be .mp4")
        exit()
    
    image_names = get_image_names(args.directory)
    if len(image_names) == 0: 
        print("Found no images")
        exit()

    keypoints, faces = load_images(image_names)
    if len(keypoints) < 2: 
        print("Found", len(keypoints), "faces. Not enough for morphing")
        exit()

    frames = make_transitions(keypoints, faces)
    video_util.vidwrite_from_numpy(args.output, frames[:, :, :, [2, 1, 0]])