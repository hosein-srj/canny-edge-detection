# canny-edge-detection
Edge detection is an image processing technique to find boundaries of objects in images. Here are an example image and the detected edges:

![image](https://user-images.githubusercontent.com/54143711/137169435-0d6c25af-7b6f-437f-9f2c-ca918a76c19b.png)

There are lots of edge detection algorithms but in this assignment, you will implement an algorithm with the following three steps:

• Smoothing <br>
• Gradient calculation <br>
• Non-maximum suppression <br>
• Thresholding <br>
• hysteresis thresholding <br>



<h3> Smoothing </h3>

Before starting the actual edge detection, we first smooth the image to reduce the amount of edges detected from the noise. Smoothing is typically done by applying a low-pass filter to the image (as the noise is often high frequency term in the image). In this assignment, we will use a Gaussian blur. The following example shows the effect of the Gaussian smoothing:


![image](https://user-images.githubusercontent.com/54143711/137169611-e126b56c-2355-4c8b-8aff-dbf1cdfe0e18.png)

Gaussian blur is defined as a convolution between the image and the following 5x5 filter, a discrete Gaussian function in 2D space truncated after two tabs:

![image](https://user-images.githubusercontent.com/54143711/137169652-db954414-8a56-4dce-a439-a84a5cdda211.png)

Gradient calculation

Simply put, the edges in an image are from the sudden changes of brightness, assuming that each object consists of pixels that do not greatly vary in their brightness. One way to measure how big the change is to calculate the gradient magnitude at each pixel. The gradient operator we are going to use in this assignment is Sobel operator. Sobel operator is based on the following two 3x3 filters, which calculate x and y component of the gradient, respectively:

![image](https://user-images.githubusercontent.com/54143711/137169746-93390eba-7ad8-4a34-bf6f-61fb0fdff9fd.png)

Once x and y components of the gradient is computed, the magnitude can be computed by:

![image](https://user-images.githubusercontent.com/54143711/137169811-02e8a1d6-50b1-44d3-b032-2c125edd2844.png)

result of gradient calculation:

![image](https://user-images.githubusercontent.com/54143711/137169876-8aff34dc-84b3-450a-81fe-a1b159f6fe50.png)


Non-maximum suppression

One of the artifacts from smoothing is that it also makes the object boundaries blurry. As a result, the gradients calculated from the previous step might make the final edges span multiple pixels. In order to get sharper edges, we find local maxima in the gradient image and remove the others. Local maxima can be found in various ways, but in this assignment we will implement a simple one: a gradient is considered locally maximal if it is either greater than or equal to its neighbors in the positive and negative gradient direction.

Thresholding

Once we have computed a measure of edge strength (typically the gradient magnitude), the next stage is to apply a threshold, to decide whether edges are present or not at an image point. The lower the threshold, the more edges will be detected, and the result will be increasingly susceptible to noise and detecting edges of irrelevant features in the image. Conversely a high threshold may miss subtle edges, or result in fragmented edges


hysteresis thresholding

It uses two threshold values: a high (Th) and a low (Tl) Threshold. All pixels with gradient magnitude are greater than high threshold are considered as edge pixels, and all pixels with magnitude gradient values less than low threshold are considered as noisy edge points and dicarded. Finally, the pixels with magnitude gradient value between thetwo thresholds are considered as edge point only if they are connected to strong edge points.


Final Result Of Edge detection by canny:

![final_edges](https://user-images.githubusercontent.com/54143711/137170207-b886ef5a-dc45-449a-965d-e06cb629b1f9.jpg)


