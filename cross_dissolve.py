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

    # Disassemble frequently used variables
    keypoints1 = im1interp[0][0]
    simplex1 = im1interp[0][1]
    keypoints2 = im2interp[0][0]
    simplex2 = im2interp[0][1]

    # Store warped images at each point
    warped_images = np.zeros((num_images, num_interpolations + 1, *image_size))
    warped_mask = np.zeros((num_images, num_interpolations + 1, *image_size))

    # Set initial and end images
    warped_images[0, 0] = cropped_images[0]
    warped_mask[0, 0] = np.ones(warped_mask[0, 0].shape)
    warped_images[1, -1] = cropped_images[1]
    warped_mask[1, -1] = np.ones(warped_mask[1, -1].shape)

    # Determine what the warping looks like at each timestep
    for time_idx in range(1, num_interpolations + 1): 
        warped_images[0, time_idx], warped_mask[0, time_idx] = warping.get_morphed_image(keypoints1, im1interp[time_idx][0], simplex1, cropped_images[0])
        warped_images[1, num_interpolations-time_idx], warped_mask[1, num_interpolations-time_idx] = warping.get_morphed_image(keypoints2, im2interp[time_idx][0], simplex2, cropped_images[1])

    # Cross dissolve warpings at each time step
    image_progression = np.zeros((num_interpolations + 1, *image_size))
    factor = 1 / num_interpolations
    contributions = warped_mask.sum(axis=0)

    # Try to fill holes with contents from other image
    warped_images[0] = np.logical_and(warped_mask[0] == 0, contributions > 0)  * warped_images[1] + (warped_mask[0] == 1) * warped_images[0]
    warped_images[1] = np.logical_and(warped_mask[1] == 0, contributions > 0) * warped_images[0] + (warped_mask[1] == 1) * warped_images[1]

    # Set morphed image at different time periods to be weighted averages of warped images
    for time_idx, image in enumerate(warped_images[0]): 
        image_progression[time_idx] = image * (1-time_idx*factor)
        
    for time_idx, image in enumerate(warped_images[1]): 
        image_progression[time_idx] += image * time_idx * factor
    
    # Set places where both images don't effect color to black
    image_progression[np.where(contributions == 0)] = 0

    return image_progression.astype(cropped_images[0].dtype) # (contributions * 255/2).astype(cropped_images[0].dtype)[:, 300:400, 150:300, :]
        