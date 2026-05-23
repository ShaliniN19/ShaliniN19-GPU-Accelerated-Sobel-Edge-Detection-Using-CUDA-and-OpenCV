# GPU-Accelerated-Sobel-Edge-Detection-Using-CUDA-and-OpenCV


## DESCRIPTION:

This project implements Sobel Edge Detection using CUDA and OpenCV for GPU-accelerated image processing. The program reads a grayscale image, applies the Sobel filter in parallel using CUDA threads, and detects image edges efficiently. The processed output image is saved after measuring GPU execution time.

## PROGRAM:
```
%%writefile sobelEdgeDetectionFilter.cu

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage,
                            unsigned char *dstImage,
                            unsigned int width,
                            unsigned int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 &&
        y > 0 && y < height - 1) {

        int gx = 0;
        int gy = 0;

        gx = -srcImage[(y - 1) * width + (x - 1)]
             -2 * srcImage[y * width + (x - 1)]
             -srcImage[(y + 1) * width + (x - 1)]
             +srcImage[(y - 1) * width + (x + 1)]
             +2 * srcImage[y * width + (x + 1)]
             +srcImage[(y + 1) * width + (x + 1)];

        gy = -srcImage[(y - 1) * width + (x - 1)]
             -2 * srcImage[(y - 1) * width + x]
             -srcImage[(y - 1) * width + (x + 1)]
             +srcImage[(y + 1) * width + (x - 1)]
             +2 * srcImage[(y + 1) * width + x]
             +srcImage[(y + 1) * width + (x + 1)];

        int magnitude = sqrtf((gx * gx) + (gy * gy));

        if (magnitude > 255)
            magnitude = 255;

        dstImage[y * width + x] = (unsigned char)magnitude;
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n",
                cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {

    Mat image = imread("/content/pca.jpg",
                       IMREAD_GRAYSCALE);

    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t imageSize =
        width * height * sizeof(unsigned char);

    unsigned char *h_outputImage =
        (unsigned char *)malloc(imageSize);

    unsigned char *d_inputImage,
                  *d_outputImage;

    checkCudaErrors(cudaMalloc(&d_inputImage,
                               imageSize));

    checkCudaErrors(cudaMalloc(&d_outputImage,
                               imageSize));

    checkCudaErrors(cudaMemcpy(d_inputImage,
                               image.data,
                               imageSize,
                               cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize(16, 16);

    dim3 gridSize((width + 15) / 16,
                  (height + 15) / 16);

    cudaEventRecord(start);

    sobelFilter<<<gridSize, blockSize>>>(
        d_inputImage,
        d_outputImage,
        width,
        height);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds,
                         start,
                         stop);

    checkCudaErrors(cudaMemcpy(h_outputImage,
                               d_outputImage,
                               imageSize,
                               cudaMemcpyDeviceToHost));

    Mat outputImage(height,
                    width,
                    CV_8UC1,
                    h_outputImage);

    imwrite("/content/output_sobel.jpeg",
            outputImage);

    free(h_outputImage);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Total time taken: %f milliseconds\n",
           milliseconds);

    return 0;
}
```

```

import cv2
from matplotlib import pyplot as plt

output_image = cv2.imread(
    '/content/output_sobel.jpeg',
    cv2.IMREAD_GRAYSCALE
)

plt.imshow(output_image, cmap='gray')
plt.title('Edge Detection Output')
plt.axis('off')
plt.show()
```

## OUTPUT:
<img width="917" height="690" alt="image" src="https://github.com/user-attachments/assets/b329dce4-8e08-4ecc-9179-013ba0c4b2e2" />


## RESULT:

The Sobel edge detection algorithm was successfully implemented using CUDA. The GPU processed the input image in parallel and generated the output image with clearly highlighted edges. The execution time was measured in milliseconds, showing faster image processing performance using GPU acceleration compared to traditional CPU-based execution.
