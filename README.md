# QR Code Detector using Classical Image Processing

There are several different ways to do this using image proocessing: (assumption being we have an in-the-wild QR code and we have to get bounding box)

Perform a Canny followed by a Hough Line transform and hopefully find 4 intersecting square like structures and filtering all such quadruplets.

But, in general the process can be separated into the following steps:

1. Image pre-processing
2. Finder Pattern Detection
3. Grouping Finder Patterns
4. Extrapolate Finder points to the fourth point (assumption allows for tracking any version of QR)

There can be multiple ways we can perform each of these steps:

1. Ideally neural re-lighting methods ought to be the most robust, but a sauvola binarization works quite well and in-fact handles noise issues to some extent too. This takes care of most of our pre-processing.
2. Detecting finder patterns can be performed using multiple ways:
    1. Use the size ratio method. QR codes have been designed in a way that the ratio of black pixel blocks and white pixel blocks in the order bwbwb exists in a 1:1:3:1:1 size ratio from most viewing angles. Hence we can just use a method where we test individual pixel and if we find a horizontal bucket in the above ratio, we test the vertical bucket and hence we can find the center of the total shape by using centers of the individual lines.
    2. Calculate connected components on the image. Using the components discover components that have very nearby centroids with bounding box of the smaller box completely within the bounding box of the larger one. Using a few other constraints, we can detect the centroids and the module width effectively.
    3. Apply Canny on the image and then compute edge contours. Filter the contours to get square like structures which are the exact object aligned bounding boxes of the finder patterns.
3. Grouping is a simple operation which essentially constructs right angled triangles with detected finder patterns and could potentially use a line scanning algorithm to find existence of a quiet space around the 2 sides as well as parse timing patterns and evaluate them statistically.
4. Finding the fourth point is another complex stage in the algorithm:
    1. A pre-processing step here is to align the 3 finder patterns to the adjacent outer corners. I have tried using Harris Corner detection and applied a distance based point eliminiation. But, this approach is highly prone to even slighly problematic gradients (Tried Canny based on median based parameter setting, but results very highly unstable).
    2. There are other methods to align these points to the corners, including classing serial processing, like getting neighborhood projected edge detections and scoring the neighborhood based on standard deviation, while moving the desired points.
    3. Another method is to calcuate the above directional projected edge detections and finding the local maxima and minima. The assumption is iif the window is equal to module width, then the maxima and minima corresponds to the outer border and the inner borders respectively. Using this we can push the corner point for each direction and align to the actual corner.
    4. The above methods do not really matter when we have cylindrical transformation present in the real life object, hence I have chosen to ignore them and rely on temporal smoothing instead (described later).
    5. The only processing I do in this step, is to additionally calculate Finder points extensions. As this is a right triangle, we can extend accross each side to find either the corner point or a point on some side in the middle of the finder pattern. Using these we can set the corner points quite close the actual corner, while the side points aid in finding line equations so we can intersect them to find the fourth point. This generally works better on most perspective transformations when compared to a direct parallelogram assumption.
    6. Ideally, we should optimize each such line I have found in the above point which is implemented. So we should calcualte the intersection of the line with the QR code itself and make use of the quiet zone principle to alter the slope of the line, keeping one of the points fixed. Optimization, could come in the form of pre-defined slope variants for each line and an arithemtic optimization to find the slopes that provide a consistent fourth point, while reducing the intersection and non-intersection score. (Feel like there could be a more continuous optimization scheme here, but step wise discrete optimization must work too)

Using the above techniques (I have mentioned which ones I have implemented), I was able to achieve ~20 fps. The primary computation is the detection of the finer points. But, the resuls were quite jittery prone to minimal tuning I was able to do. But, more often than not, if the QR code is within a certain distance and disttortion, we are able to detect it accurately. Hence, what I have opted to do is perform a Kalman Filter and produce smoothed outputs. (Note: I use parameters so that the filter aggressively adapts to new positions in new frames and not lag behind). In the demo, the green box is the smoothed bounding box, while the blue one is the currrent frame bounding box.

## QR Code detection when we have an existing template with perspective transformations

1. Lighting based preprocessing and noise filtering
2. Extract features from both the template as well as the real-life image (SURF or SIFT or ORB as we deal with perspective transformations)
3. Using a feature matching algorithm find matching points and filter matches based on least distances.
4. Compute the homography using the matched points.
5. Using the inverse transformation, we can transform the edges of the template qr code into the real-life image space and then we just draw the bounding box.
