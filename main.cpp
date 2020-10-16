/*
    Head tracking using webcam and colored blobs
    Interface with Virtual Joystick

    Created by Calvin Ng, October 2020
    Released under MIT License

    Uses OpenCV
    License: https://opencv.org/license/
*/
#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<future>
#include<mutex>
#include<chrono>
#include<thread>

#define SPLIT_COLS 3
#define SPLIT_ROWS 2
#define NUM_NOISE_FRAMES 11 // must be odd

using namespace std;

mutex mtx;
volatile int global_threshold_upper = 170;
volatile int global_threshold_lower = 130;
volatile int global_blob_size = 15;
const int global_frame_height = 120;
const int global_frame_width = 160;

void mouse_callback(int  event, int  x, int  y, int  flag, void *param)
{
    if (event == cv::EVENT_MOUSEMOVE) {
        cv::Point *c = (cv::Point *) param;
        c->x = x;
        c->y = y;
        return;
    }
}

int cvHandler()
{
    cv::Point mouseCoords(0,0);

    cv::VideoCapture camera(0);
    if(!camera.isOpened())
    {
        cerr << "ERROR: Could not open camera" << endl;
        return 1;
    }

    camera.set(cv::CAP_PROP_FRAME_WIDTH, global_frame_width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, global_frame_height);

    cout<<"resolution is: "<<global_frame_width<<" x "<<global_frame_height<<endl;


    //preview windows
    cv::namedWindow("Hue", cv::WINDOW_NORMAL);
    cv::resizeWindow("Hue", 320, 240);
    cv::setMouseCallback("Hue", mouse_callback, &mouseCoords);

    cv::namedWindow("Filtered", cv::WINDOW_NORMAL);
    cv::resizeWindow("Filtered", 320, 240);

    cv::namedWindow("Color", cv::WINDOW_NORMAL);
    cv::resizeWindow("Color", 320, 240);

    cv::namedWindow("Process", cv::WINDOW_NORMAL);
    cv::resizeWindow("Process", 320, 240);

    cv::namedWindow("Keypoints", cv::WINDOW_NORMAL);
    cv::resizeWindow("Keypoints", 320, 240);

    auto startTime = chrono::system_clock::now();

    //  these 2 vectors have to persist between iterations
    vector<cv::Mat> rawFrames;
    vector<cv::Point> targets;
    while (1)
    {
        //  capture an image and crop
        cv::Mat colorFrame;
        camera>>colorFrame;
        cv::Rect cropRect(0, global_frame_height/2, global_frame_width, global_frame_height/2);
        colorFrame = colorFrame(cropRect);

        //  split into hue, saturation, value
        cv::Mat hsvFrame, hueFrame, saturationFrame, valueFrame;
        vector<cv::Mat> hsvChannels;

        cv::cvtColor(colorFrame, hsvFrame, cv::COLOR_BGR2HSV);
        cv::split(hsvFrame, hsvChannels);
        hueFrame = hsvChannels[0];
        saturationFrame = hsvChannels[1];
        valueFrame = hsvChannels[2];

        //  build a vector of images based on NUM_NOISE_FRAMES (number of frames used in denoising)
        //  we only use the hueFrame because it is invariant with respect to lighting (luminance)
        int frameVectorSize = (int) rawFrames.size();
        if(frameVectorSize >= NUM_NOISE_FRAMES)
        {
            rawFrames.pop_back();
        }
        rawFrames.emplace(rawFrames.begin(), hueFrame.clone());

        //  conditionally apply denoising, only if there's enough images in the vector, else just use the latest image
        cv::Mat filteredFrame;
        if(frameVectorSize >= 3 && frameVectorSize%2 != 0)
        {
            //  fastNlMeansDenoisingMulti( input array, output, index of image to filter, (n) images to process, filter strength, windowSize, searchWindowSize)
            //  NOTE: time complexity is O(searchWindowSize) + O(n)
            //  we are filtering the image at the middle of the vector, using all the images before and after
            cv::fastNlMeansDenoisingMulti(rawFrames, filteredFrame, frameVectorSize/2, frameVectorSize, 20, 7, 11);
        }
        else
        {
            filteredFrame = rawFrames.back();
        }

        /* apply thresholds to the image, selecting only pixels with the desired hue

            threshold types:
            0: binary
            1: binary inverted
            2: threshold truncate
            3: threshold to zero
            4: threshold to zero inverted
        */

        int lowerThreshold, upperThreshold;


        //  safely obtain the threshold values from another thread using mutex
        mtx.lock();
        lowerThreshold = global_threshold_lower;
        upperThreshold = global_threshold_upper;
        mtx.unlock();

        cv::Mat newFrame;
        cv::threshold(filteredFrame, newFrame, lowerThreshold, 255, 3); //  all pixels with hue below the lower threshold are black
        cv::threshold(newFrame, newFrame, upperThreshold, 255, 4);      //  all pixels with hue above the upper threshold are black
        cv::threshold(newFrame, newFrame, 1, 255, 0);                   //  remaining pixels become pure white

//        //  configure the blob detector
//        cv::SimpleBlobDetector::Params params;
//        params.filterByArea = true;
//
//        // safely obtain the blob size from another thread using mutex
//        mtx.lock();
//        params.minArea = global_blob_size;
//        mtx.unlock();
//
//        params.filterByInertia = false;
//        params.filterByConvexity = false;
//        params.filterByCircularity = false;
//        params.filterByColor = false;

        //  detect the contours
        vector<vector<cv::Point>> contours;
        cv::findContours(newFrame, contours, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_TC89_L1);

        // iterate over the list of contours to find the 2 biggest areas
        vector<vector<cv::Point>> biggestContours;
        if(contours.size() >= 2)
        {
            if(cv::contourArea(contours[0]) > cv::contourArea(contours[1]))
            {
                biggestContours.push_back(contours[0]);
                biggestContours.push_back(contours[1]);
            }
            else
            {
                biggestContours.push_back(contours[1]);
                biggestContours.push_back(contours[0]);
            }

            for(int i = 2; i < contours.size(); i++)
            {
                //  if the current contour is bigger than the one at the beginning of biggestContours (it's bigger than the biggest)
                if(cv::contourArea(contours[i]) > cv::contourArea(biggestContours.front()))
                {
                    biggestContours.emplace(biggestContours.begin(), contours[i]);
                    biggestContours.pop_back();
                }
                //  if the current contour is bigger than the one at the end of biggestContours (it's bigger than the 2nd biggest)
                else if(cv::contourArea(contours[i]) > cv::contourArea(biggestContours.back()))
                {
                    biggestContours.emplace(biggestContours.begin() + 1, contours[i]);
                    biggestContours.pop_back();
                }
            }
        }

        //  generate a color image to draw the contour outlines
        cv::Mat frameWithContours(newFrame.size(), CV_8UC3);
        cv::cvtColor(newFrame, frameWithContours, cv::COLOR_GRAY2BGR);
        cv::drawContours(frameWithContours, biggestContours, -1, cv::Scalar(255, 0, 0));

        //  draw a circle at mouse cursor
        cv::circle(hueFrame, mouseCoords, 5, cv::Scalar(255, 0, 0));

        // debugging text
        cout<<mouseCoords.x<<", "<<mouseCoords.y<<": ";
        cout<<upperThreshold<<"u "<<lowerThreshold<<"l ";
        cout<<contours.size()<<"b ";

        //  display the current hue under the mouse cursor
        int cursorValue;
        cursorValue = (int)hueFrame.at<uchar>(mouseCoords);
        cout<<cursorValue<<", ";

        //  preview windows
        cv::imshow("Hue", hueFrame);
        cv::imshow("Filtered", filteredFrame);
        cv::imshow("Color", colorFrame);
        cv::imshow("Process", newFrame);
        cv::imshow("Keypoints", frameWithContours);

        auto timeNow = chrono::system_clock::now();
        auto frameDuration = chrono::duration_cast<chrono::milliseconds>(timeNow - startTime);
        int frameMs = frameDuration.count();
        startTime = timeNow;
        cout<<(1000./frameMs)<<" FPS"<<endl;

        // wait (5ms) for a key to be pressed (exit)
        if (cv::waitKey(5) >= 0)
        {
            break;
        }

        //this_thread::sleep_for(100ms);
    }
    return 0;
}

int main(int argc, char *argv[])
{
    auto cvThread = async(cvHandler);

    char type;
    int newThreshold;

    while(cvThread.wait_for(chrono::milliseconds(100)) == future_status::timeout)
    {
        cout<<"ready";
        cin>>type>>newThreshold;

        if(type == 'u')
        {
            mtx.lock();
            global_threshold_upper = newThreshold;
            mtx.unlock();

            cout <<"new threshold"<<newThreshold;
        }
        else if(type == 'l')
        {
            mtx.lock();
            global_threshold_lower = newThreshold;
            mtx.unlock();

            cout <<"new threshold"<<newThreshold;
        }
        else if(type == 'b')
        {
            mtx.lock();
            global_blob_size = newThreshold;
            mtx.unlock();
        }
    }
    cout<<"done";
    return 0;
}

