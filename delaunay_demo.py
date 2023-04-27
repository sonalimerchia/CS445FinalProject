import cv2
import matplotlib.pyplot as plt
from delaunay import create_triangulations

image = cv2.cvtColor(cv2.imread('img_341.jpg'), cv2.COLOR_BGR2GRAY)
triangulations = create_triangulations(image)

for feature_points, triangulation in triangulations:
    plt.figure()
    plt.imshow(image, cmap='gray')
    for (x, y) in feature_points:
        plt.plot(x, y, 'ro')
    plt.triplot(feature_points[:, 0], feature_points[:, 1], triangulation)
    plt.show()