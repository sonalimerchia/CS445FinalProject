# CS445FinalProject

To run the `.ipynb` file, be sure to unzip `shape_predictor_68_face_landmarks.dat.bz2` first.

Make sure you have the following python modules installed:

1. OpenCV (`cv2`)
2. NumPy (`numpy`)
3. matplotlib
4. SciPy (`scipy`)
5. dlib
6. ffmpeg

Then you can generate a video by passing in a folder path and output video file name as arguments like so:
`python3 morphing_video.py --directory=images/ --output=video.mp4`

[Here](https://drive.google.com/file/d/1MIY1yyhvu0xkoQh-2VsyL1RsPyRdcnFG/view?usp=share_link) is an example video made with 392 faces from the `00` portion of the [images dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) scraped off wikipedia.
