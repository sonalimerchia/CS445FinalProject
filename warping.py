from matplotlib.tri import Triangulation
import cv2 
import numpy as np 

def get_transformations(keypoints1, keypoints2, simplexes): 
    '''
    Input: 
        keypoints1: the points you are transforming from [num_points, 2]
        keypoints2: the points you are transforming to [num_points, 2]
        simplexes: the simplexes for the points [num_simplexes, 3]

    Output: 
        [num_simplexes, 2, 3] array where each element at index i is the 
            the transform you need to apply to barycentric coordinates of the triangle represented
            by the ith simplex to get the barycentric coordinates of the same triangle in keypoints2
    '''

    num_simplexes = len(simplexes)
    Ts = np.zeros((num_simplexes, 2, 3))

    for simplex_idx, simplex in enumerate(simplexes): 
        p0s = keypoints1[simplex]
        p1s = keypoints2[simplex]

        p0s = (p0s - p0s.min(axis=0)).astype(np.float32)
        p1s = (p1s - p1s.min(axis=0)).astype(np.float32)

        Ts[simplex_idx] = cv2.getAffineTransform(p0s, p1s)

    return Ts

def get_masks(points, simplexes, dimensions, dtype): 
    '''
    Input: 
        points: points that correspond to a feature identification [num_points, 2]
        simplexes: indices of the points that correspond to a triangulation [num_simplexes, 3]
        dimensions: (height of image, width of image)
        dtype: the type of the numpy array of the image 
    Output: 
        A tuple with the following elements: 
        0: An array of masks [num_simplexes, height of image, width of image] such that the 
            i^th element is the mask that (when applied to the image), will only show the 
            pixels falling within the i^th simplex
        1: An array of the same size as the image such that each element corresponds to the index
            of the simplex the pixel belongs to
    '''
    
    num_simplexes = simplexes.shape[0]
    masks = np.zeros((num_simplexes, *dimensions), dtype=dtype)

    tri =  Triangulation(points[:, 1], points[:, 0], triangles=simplexes)
    [Ys, Xs] = np.indices(dimensions)

    simplex_grid = np.array(tri.get_trifinder()(Ys, Xs))

    for i in range(num_simplexes): 
        masks[i] = simplex_grid == i

    return masks, simplex_grid

def get_bounding_box(points, simplexes, H, W): 
    '''
    Input: 
        points: points in an image [num_points, 2]
        simplexes: indices of the points that correspond to a triangulation [num_simplexes, 3]
        H: the height of the image
        W: the width of the image
    Output: 
        A tuple with the following elements: 
        0: the top left corner of each simplex [num_simplexes, 2]
        1: the bottom right corner of each simplex [num_simplexes, 2]
    '''
    triangles = points[simplexes]
    topleft = triangles.min(axis=1)

    # Add 1 to handle minor discrepancies between matplotlib triangulation and cv2 affine transformation
    bottomright = triangles.max(axis=1) + 1

    # Make sure the coordinates fall within the image
    # (might not due to rounding issues)
    topleft[:, 0] = np.clip(topleft[:, 0], 0, H)
    bottomright[:, 0] = np.clip(bottomright[:, 0], 0, H)

    topleft[:, 1] = np.clip(topleft[:, 1], 0, W)
    bottomright[:, 1] = np.clip(bottomright[:, 1], 0, W)

    return topleft, bottomright

def get_morphed_image(initial_pts, interpolated_pts, simplexes, image): 
    '''
    Input: 
        initial_pts: the points from the initial image [num_points, 2]
        interpolated_pts: the points at the time step you are generating an image for [num_points, 2]
        simplexes: the simplexes for the triangular mesh [num_simplexes, 3]
        image: the image to be morphed [image_height, image_width] or [image_height, image_width, color_channels]
    Output: 
        A tuple with the following elements: 
        0: a warped image of the same size where the contents of each triangle in the mesh are the
            affine transformations of the corresponding triangles in the initial images.
        1: a mask such that [x, y] is 1 if some point in the warping contributes to that pixel
    '''
    H, W = image.shape[0], image.shape[1]
    num_simplexes = simplexes.shape[0]

    # Round points so they're valid indexes
    initial_pts = np.round(initial_pts).astype(np.int64)
    interpolated_pts = np.round(interpolated_pts).astype(np.int64)

    # Determine affine transformations from initial points to interpolated points
    Ts = get_transformations(initial_pts, interpolated_pts, simplexes)
    masks, _ = get_masks(initial_pts, simplexes, (H, W), image.dtype)
    
    # Make mask work if more than 1 color channel
    if len(image.shape) > 2: 
        C = image.shape[2]
        masks = np.stack((masks, masks, masks), axis=3)

    output_img = np.zeros(image.shape)
    output_mask = np.zeros(image.shape)

    # Determine bounding boxes of triangles
    topleft_inits, bottomright_inits = get_bounding_box(initial_pts, simplexes, H, W)
    topleft_final, bottomright_final = get_bounding_box(interpolated_pts, simplexes, H, W)

    # Perform affine transformations
    for simplex_idx in range(num_simplexes): 
        [y1, x1] = topleft_inits[simplex_idx]
        [y2, x2] = bottomright_inits[simplex_idx]
        [yp1, xp1] = topleft_final[simplex_idx]
        [yp2, xp2] = bottomright_final[simplex_idx]

        if x2-x1 <= 0 or y2-y1 <= 0 or yp2-yp1 <= 0 or xp2-xp1 <= 0: 
            continue

        input_image = image[x1:x2, y1:y2]
        input_mask = masks[simplex_idx][x1:x2, y1:y2]
        warped_image = cv2.warpAffine(input_image, Ts[simplex_idx], (yp2-yp1, xp2-xp1))
        warped_mask = cv2.warpAffine(input_mask, Ts[simplex_idx], (yp2-yp1, xp2-xp1))

        xp2 = min(xp2, xp1 + output_img[xp1:xp2].shape[0])
        yp2 = min(yp2, yp1 + output_img[xp1:xp2][yp1:yp2].shape[1])

        w = xp2-xp1
        h = yp2-yp1

        if w == 0 or h == 0: 
            continue
        
        output_img[xp1:xp2, yp1:yp2] += warped_image[:w, :h] * warped_mask[:w, :h]
        output_mask[xp1:xp2, yp1:yp2] += warped_mask[:w, :h]

    output_mask = np.clip(output_mask, 0, 1)

    # Return image as same type given (might be given as uint and averaging messed it up)
    return output_img.astype(image.dtype), output_mask