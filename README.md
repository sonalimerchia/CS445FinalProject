# CS445 Final Project

There is an example to look at the steps implemented by the module in `face_morphing.ipynb` and a utility to generate morphing videos from a folder of images. 

To run either of these files, be sure to download and unzip [`shape_predictor_68_face_landmarks.dat.bz2`](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2) first and make sure you run the following:

```
virtualenv venv
source venv/bin/activate

pip install -r requirements.txt
```

If you run into an issue with dlib, you will have to run `xcode-select --install` on MacOS first. On Windows, you will have to make sure CMake is in the PATH.

Then you can generate a video by passing in a folder path and output video file name as arguments like so:
`python3 morphing_video.py --directory=images/ --output=video.mp4`

[Here](https://drive.google.com/file/d/1MIY1yyhvu0xkoQh-2VsyL1RsPyRdcnFG/view?usp=share_link) is an example video made with 392 faces from the `00` portion of the [images dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) scraped off wikipedia.
