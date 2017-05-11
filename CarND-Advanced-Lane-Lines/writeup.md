## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)



[image1]: ./output_images/undist.png "Undistorted"
[image2]: ./test_images/test5.jpg "Road Transformed"
[image3]: ./output_images/pipeline.png "pipeline Example"
[image4]: ./output_images/undist_warped.png "Warp Example"
[image5]: ./output_images/line_fit.png "Fit Visual"
[image6]: ./output_images/lane_area.png "Output"
[image7]: ./output_images/output_img_in_video.png "Output"
[video1]: ./output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./AdvancedLaneFinding.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result. The first image is the test image and 

![alt text][image0]
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in the 3rd code cell of the IPython notebook.  The `warper()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[180, 720],
    [585, 455],
    [695, 455],
    [1100, 720]])
dst = np.float32(
    [[280, 760],
    [280, 0],
    [1000, 0],
    [1000, 760]])


```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 180, 720      | 280, 760        | 
| 585, 455      | 280, 0      |
| 695, 455      | 1000, 0      |
| 1100, 720     | 1000, 760       |


Then I compute the perspective transform matrix given source and destination points by calling cv2.getPerspectiveTransform() function. And then use the transform matrix to warp the image by calling cv2.warpPerspective() function.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I tried a combination of color and gradient thresholds to generate a binary image in the combined_threshold_binary function in the 5th cell. I found that thresholding the image using HSV color space's V channel (it's better than the S channel in the HLS color space!) along with the various gradient thresholds produce the best result for me. Below is what I tried you can see the final result in the last image. 

![alt text][image3]



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used sliding window technique as described in the class and fit my lane lines with a 2nd order polynomial (see the 7th cell in the python notebook), and visualization is like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the set_curvature and set_line_base_pos functions in the Line class definition in the 11th cell in the notebook. The curvature is computed using the formula as described in the class. The position of the vehicle is computed as the distance from the center of the vehicle to the (left/right) line. And later the offset is computed as the distance of the vehicle center to the left line minus the distance to the right line.

Two Line() objects are created for left and right lines. In each frame of the video, we run fit_lines() or continue_fit_lines() functions (the latter is based on the past fit line for speedup fit) to detect the left and right lines and return the fit coefficients and x pixel positions of the left and right line, which are then passed to the Line objects and used to compute the curvature and line_base_pos of the Line object as described previously.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in draw_line_area function the 10th cell in the notebook. Here is an example of my result on a test image:

![alt text][image7]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The two area I spent a lot of time on this project are 1) figuring out what's the best combination of color thresholding and gradient thresholding and their threshold values, which requires visualization and tuning the threshold values. 2) figuring out how to best warp the image to produce two parallel lane lines.

The pipeline is still sensitive to coloring of the road and light reflections of the front window shield, and other changes such as approaching/passing vehicles as evidenced by the results with the challenge videos. I think I need to experiment with the images in the challenge videos and extract what the typical problems/noises are. Next step would be to use computer vision techniques and deep learning to identify/remove the noise.
