import warping
import cv2
import numpy as np

def dissolve(im1interp, im2interp, cropped_images): 
    '''
    Input: 
        im1interp: interpolation data for the first image. Interpolation data is 
            a list of tuples where each list index is the time index, 
            the first element of the tuple is the location of the points at that time step,
            the second element of the first tuple is the simplexes calculated by the Delaunay algorithm
        im2interp: the interpolation data for the second image (described above)
        cropped_images: the cropped BGR color images of the 2 faces to transform
    Output: 
        An array where the i^th element is the cross_dissolve of the morphed images at that time step in BGR color space
    '''
    assert len(im1interp) == len(im2interp)
    assert len(cropped_images) == 2

    # Unpack variables
    num_interpolations = len(im1interp) - 1
    num_images = len(cropped_images)
    image_size = cropped_images[0].shape

    keypoints1 = im1interp[0][0]
    simplex1 = im1interp[0][1]
    keypoints2 = im2interp[0][0]
    simplex2 = im2interp[0][1]

    # Convert to LAB color space for smoother transition
    cropped_images = [cv2.cvtColor(image, cv2.COLOR_BGR2LAB) for image in cropped_images]
    warped_images = np.zeros((num_images, num_interpolations + 1, *image_size))

    # Set initial and end images
    warped_images[0, 0] = cropped_images[0]
    warped_images[1, -1] = cropped_images[1]

    # Determine what the warping looks like at each timestep
    for time_idx in range(1, num_interpolations + 1): 
        warped_images[0, time_idx] = warping.get_morphed_image(keypoints1, im1interp[time_idx][0], simplex1, cropped_images[0])
        warped_images[1, num_interpolations-time_idx] = warping.get_morphed_image(keypoints2, im2interp[time_idx][0], simplex2, cropped_images[1])

    # Cross dissolve warpings at each time step
    image_progression = np.zeros((num_interpolations + 1, *image_size))
    factor = 1 / num_interpolations
    for time_idx, image in enumerate(warped_images[0]): 
        image_progression[time_idx] = image * (1-time_idx*factor)

    for time_idx, image in enumerate(warped_images[1]): 
        image_progression[time_idx] += image * time_idx * factor
    
    # Convert back to BGR
    for time_idx, image in enumerate(image_progression): 
        image_progression[time_idx] = cv2.cvtColor(image.astype(cropped_images[0].dtype), cv2.COLOR_LAB2BGR)

    return image_progression.astype(cropped_images[0].dtype)
        