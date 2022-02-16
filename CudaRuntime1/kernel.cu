/*
Author: Mike Filburn
Date: 2/1/22
Description:
    Program takes n photos and converts to grayscale using the first or default camera. It then gets those images (usong GPU) and gets an average per 
    pixel. This produces a slow, but very clean grayscale image. Once the image is produced It is blurred and edges are produced. The image is converted
    to B/W so the contours are value independent. Finally the contours are found.

    Then all the contours are loaded onto the GPU and they are all evaluated by size. If the size is to small or not will be stored in an array of 
    essentially boolean values. The array is then returned back to the CPU side then only the contours between min/max will be displayed. There are 2 sliders
    that can change the min/max real time.

    Note: any keypress will end the program
*/


#include <iostream>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


/* 
NOTES For CUDA Kernal below :
 int *dev_heap_array is the memory address on the gpu where the heap is stored
 int *dev_reassembled_array is the memory address of wheret the finished calculated image will go.
 int size is the size of the Mat (W x H) or 640 x 480 in laptops camera case.
 int samples is the number of images taken in the buffer step.
*/
__global__ void addKernel(int *dev_heap_array, int *dev_reassembled_array, int size_count, int samples)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size_count)
    {
        int temp = 0;

       for (int i = 0; i < samples; i++)
       {
           temp += dev_heap_array[i + x * samples];
       }

       temp = temp / samples;

       dev_reassembled_array[x] = temp;
    }
}

__global__ void removeKernel(int contours_size, int largest, cv::Point *dev_contours, int min, int max, int *dev_good_contours_int, int *dev_good_contours_arr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < contours_size)
    {
        int size = 0;
        
        // Look for how many points are valid
        for (int i = 0; i < largest; i++)
        {
            if (dev_contours[i + (x * largest)].x != 0 && dev_contours[i + (x * largest)].y != 0)
            {
                size++;
            }
        }

        //if they are valid, save it to the array. Increment what is good..
        if (size > min && size < max)
        {
            (*dev_good_contours_int)++;
            dev_good_contours_arr[(*dev_good_contours_int)] = x;
        }
    }
}

struct mat_pack_
{
    cv::Mat image_feed;
    cv::Mat image_feed_gray;
    cv::Mat image_feed_gray_blur1;
    cv::Mat image_feed_gray_blur2;
    cv::Mat image_feed_gray_roi;
    cv::Mat sobel;
    cv::Mat sobelImage;
    cv::Mat sobelThresholded;
    std::vector<cv::Mat> heap;

    int iteration = 0;

    int scroll_bar_a = 100;
    int scroll_bar_b = 300;
};

void experiment(mat_pack_);

const int samples = 10;  //larger samples = better quality but slows computer.

int main()
{
    // Initilize Camera return if fails.
    cv::VideoCapture cap_from_cam(0);
    if (!cap_from_cam.isOpened())
    {
        std::cout << "No camera found. Sorry man.";
        return -1;
    }

    //create a window
    cv::namedWindow("MTG-OCR", cv::WINDOW_AUTOSIZE);

    mat_pack_ mat_pack;

    //Create Sliders
    cv::createTrackbar("Min", "MTG-OCR", &mat_pack.scroll_bar_a, 1000);
    cv::createTrackbar("Max", "MTG-OCR", &mat_pack.scroll_bar_b, 1000);

    //produce global rows, cols, samples here.
    cv::Mat cam_sample;
    cap_from_cam >> cam_sample;
    cv::cvtColor(cam_sample, cam_sample, cv::COLOR_BGR2GRAY);

    int row_count = cam_sample.rows;          //normally default 480 for laptop cam
    int col_count = cam_sample.cols;          //normally default 640 for laptop cam
    int size_count = row_count * col_count;   //total screne 640 * 480 for laptop cam

    //make one large array on the heap to hold all the data to be analysed.
    int* heap_array = new int[size_count * samples];    //massive array that is w * h * number of pictures. so 10 samples = 640 * 480 * 10 = 3,072,000
    int* reassembled_array = new int[size_count];       //smaller array that is just w * h to hold the reconstructed MAT element values when on the GPU


    //Fill the massive heap with N photos. Just done once for initilization.
    for (int i = 0; i < samples; i++)
    {
        cv::Mat temp;
        cap_from_cam >> temp;
        cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);

        for (int k = 0; k < temp.rows; k++)
        {
            for (int j = 0; j < temp.cols; j++)
            {
                heap_array[i + (k * col_count * samples) + (j * samples)] = (int)temp.at<uchar>(k, j);
            }
        }
    }

    int* dev_heap_array;          //just a pointer to be used on the GPU
    int* dev_reassembled_array;   //just a pointer to be used on the GPU

    //allicate memory on the GPU to be used to hold the input array and return array.
    cudaMalloc((void**)&dev_heap_array, size_count * samples * sizeof(int));
    cudaMalloc((void**)&dev_reassembled_array, size_count * sizeof(int));

    // cycle through the memory and overwrite the oldest. This is needed since pop/push dont work and its nicer to just re-use memory.
    int iteration = 0;

    // Any button will cause this to close
    while (1)
    {
        // rotate which part of the array to fill. replaces the oldest iteratioin. can't pop/push
        // should always be 0 to samples - 1;
        if (iteration >= samples)
        {
            iteration = 0;
        }
        else
        {
            iteration++;
        }

        // take a picture and change to gray
        cv::Mat cap_image;
        cv::Mat cap_image_color;
        cap_from_cam >> cap_image;
        cap_image.copyTo(cap_image_color);

        cv::cvtColor(cap_image, cap_image, cv::COLOR_BGR2GRAY);

        //store the new picture in the iteration that has been filled heap array above.
        for (int k = 0; k < row_count; k++)
        {
            for (int j = 0; j < col_count; j++)
            {
                heap_array[iteration + (k * col_count * samples) + (j * samples)] = (int)cap_image.at<uchar>(k, j);
            }
        }

        //copy the huge heap array and reassemble array to the GPU
        cudaMemcpy(dev_heap_array, heap_array, size_count * samples * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_reassembled_array, reassembled_array, size_count * sizeof(int), cudaMemcpyHostToDevice);

        //run the cuda kernal
        addKernel <<<size_count / 256 + 1, 256>>> (dev_heap_array, dev_reassembled_array, size_count, samples);

        //waits for the kernal to finish.
        cudaDeviceSynchronize();

        //copy the memory back
        cudaMemcpy(reassembled_array, dev_reassembled_array, size_count * sizeof(int), cudaMemcpyDeviceToHost);

        //get the gpu calculated pixel average and put it into a MAT pixel by pixel.
        for (int k = 0; k < row_count; k++)
        {
            for (int j = 0; j < col_count; j++)
            {
                cap_image.at<uchar>(k, j) = (uchar)reassembled_array[(k * col_count) + (j)];
            }
        }

        //display the averaged image.
        cv::imshow("MTG-OCR", cap_image);

        //blur and filter the image to assiste in finding contours.
        cv::medianBlur(cap_image, mat_pack.image_feed_gray_blur1, 5);
        bilateralFilter(mat_pack.image_feed_gray_blur1, cap_image, 3, 19, 3);
        bilateralFilter(cap_image, mat_pack.image_feed_gray_blur1, 3, 19, 3);
        bilateralFilter(mat_pack.image_feed_gray_blur1, cap_image, 3, 19, 3);
        //produce a b/w image of only the found sobel edges.
        Sobel(cap_image, mat_pack.image_feed_gray_blur1, CV_16S, 1, 0); //produce gradient in x direction
        Sobel(cap_image, mat_pack.image_feed_gray_blur2, CV_16S, 0, 1); //produce gradient in y direction
        mat_pack.sobel = abs(mat_pack.image_feed_gray_blur1) + abs(mat_pack.image_feed_gray_blur2);
        double sobmin, sobmax;
        cv::minMaxLoc(mat_pack.sobel, &sobmin, &sobmax);
        mat_pack.sobel.convertTo(mat_pack.sobelImage, CV_8U, -255. / sobmax, 255);

        // gets all the pixels outside threshold and turns them black.
        cv::adaptiveThreshold(mat_pack.sobelImage, mat_pack.sobelThresholded, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 3, 1);

        //find the contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mat_pack.sobelThresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        cv::imshow("Sobel Image", mat_pack.sobelThresholded);

        //needed to prevent crashing in the event no contours were found.
        if (!contours.empty())
        {
            //Finds and holds the largest 'int' number of the inner vectors of the outer vector. IE flattens for processing on the GPU
            // NOTE: Can speed up here if use sorting algorithm !!!
            unsigned int largest = 0;
            for (int i = 0; i < contours.size(); i++)
            {
                if (contours[i].size() > largest)
                {
                    largest = contours[i].size();
                }
            }

            //***note: cv::Point = size of 8 bytes which is = to a "long long" which is pretty huge.***

            // create the memory on the CPU side and initalize to 0
            cv::Point* flattened_contours = (cv::Point*)calloc(contours.size() * largest, sizeof(cv::Point));

            //add the vector points to the blank memory locations?
            for (int i = 0; i < contours.size(); i++)
            {
                std::memcpy(flattened_contours + (i * largest), &contours[i][0], contours[i].size() * 8);
            }

            // create the memory on the GPU side
            cv::Point* dev_contours;
            cudaMalloc((void**)&dev_contours, contours.size() * largest * sizeof(cv::Point));

            // copy the memory to the GPU
            cudaMemcpy(dev_contours, flattened_contours, contours.size() * largest * sizeof(cv::Point), cudaMemcpyHostToDevice);
            
            //free memory since not used anymore
            free(flattened_contours);

            //need a constant that is the size of the return good contours array. Max of 1000 contours can be found.
            //if greater than 1k contours are found your image is jibberesh anyway.
            const int cnt_arr_size = 1000;

            //create an array of the size of found contours and rill with -1.
            int good_contours_int = 0;
            int good_contours_arr[cnt_arr_size];
            std::fill(good_contours_arr, good_contours_arr + cnt_arr_size, -1);
            
            //create pointer for use on GPU
            int* dev_good_contours_int;
            int* dev_good_contours_arr;

            //create the memory for filtering contours on the GPU
            cudaMalloc((void**)&dev_good_contours_int, sizeof(int));
            cudaMalloc((void**)&dev_good_contours_arr, cnt_arr_size * sizeof(int));

            //mover the data to the GPU
            cudaMemcpy(dev_good_contours_int, &good_contours_int, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_good_contours_arr, &good_contours_arr, cnt_arr_size * sizeof(int), cudaMemcpyHostToDevice);

            //run the cuda kernal
            removeKernel <<< contours.size() / 256 + 1, 256 >> > (contours.size(), largest, dev_contours, mat_pack.scroll_bar_a, mat_pack.scroll_bar_b, dev_good_contours_int, dev_good_contours_arr);

            //sync the cpu and gpu to cpu doesn't take off while GPU is working.
            cudaDeviceSynchronize();

            // return the memory from the gpu
            cudaMemcpy(&good_contours_int, dev_good_contours_int, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&good_contours_arr, dev_good_contours_arr, cnt_arr_size * sizeof(int), cudaMemcpyDeviceToHost);

            //cleanup the memory to prevent leaks.
            cudaFree(dev_contours);
            cudaFree(dev_good_contours_int);
            cudaFree(dev_good_contours_arr);


            //draw the contours that are in the range of the min/max sliders.
            for (int i = 1; i < good_contours_int + 1; i++)
            {
                cv::drawContours(cap_image_color, contours, good_contours_arr[i], cv::Scalar(0, 255, 0), 1);
            }

        }

        //show the filtered image
        imshow("reassembled", cap_image_color);

        if (cv::waitKey(1) >= 0)
        {
            break;
        }

    }

    //cleanup
    cap_from_cam.release();

    //free up memory to be nice.
    cudaFree(dev_heap_array);
    cudaFree(dev_reassembled_array);

}