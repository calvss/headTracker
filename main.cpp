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
#include<cmath>

#define NUM_NOISE_FRAMES 11 // must be odd

using namespace std;

mutex mtx;
volatile int global_threshold_upper = 170;
volatile int global_threshold_lower = 130;
volatile int global_blob_size = 15;
volatile double global_filter_constant = 0.5;
volatile bool global_exit_flag = false;
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

    //  this vector has to persist between iterations
    vector<cv::Mat> rawFrames;

    //  angle of the centroid connecting line in radians, where 0 is pure left
    double previousAngle = M_PI/2;

    while(!global_exit_flag)
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
        cv::findContours(newFrame, contours, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_NONE);

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
            cout<<biggestContours.size()<<"size";
        }
        else
        {
            cout<<"no contours";
        }

        //  find the inscribed circles of the contours (https://stackoverflow.com/a/53648903/10835281)
        //      1. generate filled contour masks
        cv::Mat mask1(newFrame.size(), CV_8UC1, cv::Scalar(0)); //  uint8, 1 channel, image full of zeroes (black)
        cv::drawContours(mask1, biggestContours, 0, cv::Scalar(255), cv::FILLED);

        cv::Mat mask2(newFrame.size(), CV_8UC1, cv::Scalar(0)); //  uint8, 1 channel, image full of zeroes (black)
        cv::drawContours(mask2, biggestContours, 1, cv::Scalar(255), cv::FILLED);

        //      2. distance transforms
        cv::Mat dt1;
        cv::distanceTransform(mask1, dt1, cv::DIST_L2, 5, cv::DIST_LABEL_PIXEL);

        cv::Mat dt2;
        cv::distanceTransform(mask2, dt2, cv::DIST_L2, 5, cv::DIST_LABEL_PIXEL);

        //      3. max values of the distance transform
        //          the inscribed circle is at the position of max_loc with radius max_val
        double max_val1;
        cv::Point max_loc1;
        cv::minMaxLoc(dt1, nullptr, &max_val1, nullptr, &max_loc1);

        double max_val2;
        cv::Point max_loc2;
        cv::minMaxLoc(dt2, nullptr, &max_val2, nullptr, &max_loc2);

        //  compute the centroids using image moments
        cv::Moments moment1 = cv::moments(biggestContours[0]);
        cv::Moments moment2 = cv::moments(biggestContours[1]);

        cv::Point centroid1((moment1.m10 / (moment1.m00 + 1e-5)), (moment1.m01 / (moment1.m00 + 1e-5))); //have to add 1e-5 to prevent divide by 0
        cv::Point centroid2((moment2.m10 / (moment2.m00 + 1e-5)), (moment2.m01 / (moment2.m00 + 1e-5))); //have to add 1e-5 to prevent divide by 0

        //  for simplicity, centroid1 is always the bottom one
        if(centroid1.y < centroid2.y)
        {
            cv::Point swapCentroid = centroid1;
            centroid1 = centroid2;
            centroid2 = swapCentroid;
        }

        //  calculate the angle of the centroid connecting line
        double newAngle = atan2(centroid1.y - centroid2.y, centroid1.x - centroid2.x);

        //  low pass filter
        newAngle = ((1. - global_filter_constant) * previousAngle) + (global_filter_constant * (newAngle + previousAngle)/2);
        previousAngle = newAngle;

        //  generate a color image to draw the contour outlines
        cv::Mat frameWithContours(newFrame.size(), CV_8UC3); //  uint8, 3 channels
        cv::cvtColor(newFrame, frameWithContours, cv::COLOR_GRAY2BGR);
        cv::drawContours(frameWithContours, biggestContours, -1, cv::Scalar(255, 0, 0));

        //  draw the inscribed circles
        cv::circle(frameWithContours, max_loc1, max_val1, cv::Scalar(0, 255, 0));
        cv::circle(frameWithContours, max_loc2, max_val2, cv::Scalar(0, 0, 255));

        //  draw the line connecting the centroids
        cv::line(frameWithContours, centroid1, centroid2, cv::Scalar(255, 255, 255));

        cv::line(frameWithContours, cv::Point(50,0), cv::Point(50*cos(newAngle) + 50, 50*sin(newAngle)), cv::Scalar(255, 255, 255));

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
        cv::imshow("Color", mask1);
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
    double newThreshold;

    while(cvThread.wait_for(chrono::milliseconds(100)) == future_status::timeout)
    {
        cout<<"ready";
        cin>>type;

        if(type == 'q')
        {
            global_exit_flag = true;
            return 0;
        }
        else
        {
            cin>>newThreshold;
            switch (type)
            {
                case 'u':
                    mtx.lock();
                    global_threshold_upper = (int) newThreshold;
                    mtx.unlock();

                case 'l':
                    mtx.lock();
                    global_threshold_lower = (int) newThreshold;
                    mtx.unlock();

                case 'b':
                    mtx.lock();
                    global_blob_size = (int) newThreshold;
                    mtx.unlock();

                case 'a':
                    mtx.lock();
                    global_filter_constant = newThreshold;
                    mtx.unlock();
            }
        }
    }
    cout<<"done";
    return 0;
}

