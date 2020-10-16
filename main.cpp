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
#include<mutex>
#include<chrono>
#include<thread>
#include<cmath>

#include<windows.h>
#include"public.h"
#include"vjoyinterface.h"


#define NUM_NOISE_FRAMES 5 // must be odd
#define VJOYFPS 10

using namespace std;

mutex mtx;
volatile int global_threshold_upper = 170;
volatile int global_threshold_lower = 130;
volatile int global_blob_size = 15;
volatile double global_filter_constant = 0.6;
volatile double global_head_angle = M_PI/2;
volatile bool global_exit_flag = false;
volatile bool global_statistics_flag = false;
const int global_frame_height = 240;
const int global_frame_width = 320;
volatile uint32_t global_deadzone = 500;

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
        cerr<<"ERROR 01: Could not open camera"<< endl;
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
        }
        else
        {
            cerr<<"no contours";
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

        if(mtx.try_lock())
        {
            global_head_angle = newAngle;
            mtx.unlock();
        }

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

        if(global_statistics_flag)
        {
            // debugging text
            cout<<mouseCoords.x<<", "<<mouseCoords.y<<": ";
            cout<<upperThreshold<<"u "<<lowerThreshold<<"l ";
            cout<<contours.size()<<"b ";

            //  display the current hue under the mouse cursor
            int cursorValue;
            cursorValue = (int)hueFrame.at<uchar>(mouseCoords);
            cout<<cursorValue<<", ";

            cout<<(1000./frameMs)<<" FPS"<<endl;
        }

        // wait (5ms) for a key to be pressed (exit)
        if (cv::waitKey(5) >= 0)
        {
            global_exit_flag = true;
        }
    }
    return 0;
}

int vJoyHandler(unsigned int deviceID = 1)
{

//Testing if VJoy Driver is installed******************************************
    if(vJoyEnabled())
    {
        cout<<"VJD Enabled"<<endl;
        cout<<"Vendor: "<<static_cast<char *> (GetvJoyManufacturerString())<<endl;
        cout<<"Product: "<<static_cast<char *> (GetvJoyProductString())<<endl;
        cout<<"Version Number: "<<static_cast<char *> (GetvJoySerialNumberString())<<endl;
    }
    else
    {
        cerr<<"ERROR 02: No vJoyStick driver enabled"<<endl;
        return 1;
    }

//Testing if VJoy Driver is same version with DLL******************************
    unsigned short VerDll, VerDrv;
    if (DriverMatch(&VerDll, &VerDrv))
    {
        cout<<"vJoy Driver ("<<VerDrv<<") matches vJoy DLL ("<<VerDll<<"). OK!"<<endl;
    }
    else
    {
        cerr<<"WARN: vJoy Driver ("<<VerDrv<<") does not match vJoy DLL ("<<VerDll<<"). Continuing."<<endl;
    }

//Checking virtual device status***********************************************
    VjdStat status = GetVJDStatus(deviceID);

	switch (status)
	{
	case VJD_STAT_OWN:
		cout<<"vJoy device "<<deviceID<<" is already owned by this feeder."<<endl;
		break;
	case VJD_STAT_FREE:
		cout<<"vJoy device "<<deviceID<<" is free."<<endl;
		break;
	case VJD_STAT_BUSY:
		cerr<<"ERROR 03: vJoy device "<<deviceID<<" is already owned by another feeder.\nCannot continue."<<endl;
		return -3;
	case VJD_STAT_MISS:
		cerr<<"ERROR 04: vJoy device "<<deviceID<<" is not installed or disabled.\nCannot continue."<<endl;
		return -4;
	default:
		cerr<<"ERROR 05: vJoy device "<<deviceID<<" general error.\nCannot continue."<<endl;
		return -1;
	};

//Acquire VJoy Device**********************************************************
    if (AcquireVJD(deviceID))
    {
        cout<<"Successfully acquired device number "<<deviceID<<endl;
    }
    else
    {
        cerr<<"ERROR 06: Failed to acquire device number "<<deviceID<<endl;
        return -1;
    }

//Main Loop********************************************************************
    JOYSTICK_POSITION_V2 padPosition;
    int32_t previousHeadAngle = 0;
    while(!global_exit_flag)
    {
        auto loopStartTime = chrono::system_clock::now();

        //  POV hat position is an integer from 0 to 35999 (inclusive), set the value to 0xFFFFFFF for neutral
        //  represents the angle in centidegrees, with 0 at straight forward
        //  head angle is in radians with 0 at pure left

        mtx.lock();
        double headAngle = global_head_angle;
        mtx.unlock();

        int32_t headAngleCentidegrees = (int32_t)(headAngle * 18000./M_PI);

        //  deadzone and low pass filtering
        if(abs(headAngleCentidegrees - previousHeadAngle) > global_deadzone)
        {
            //cout<<"deadzone";
            headAngleCentidegrees = (int32_t)(((1. - global_filter_constant) * (double)previousHeadAngle) + (global_filter_constant * (double)(headAngleCentidegrees + previousHeadAngle)/2.));
            previousHeadAngle = headAngleCentidegrees;
        }

        //  convert headAngle into centidegrees and offset by 270 degrees
        uint32_t hatPos = (27000 + previousHeadAngle) % 36000;
        padPosition.bHats = hatPos;

        //  send gamepad state to vJoy device
        if(!UpdateVJD(deviceID, &padPosition))
		{
			cerr<<"ERROR 07: Feeding vJoy device number "<<deviceID<<" failed - try to enable device then press enter"<<endl;
			cin.get();
			if (AcquireVJD(deviceID))
            {
                cout<<"Successfully acquired device number "<<deviceID<<endl;
            }
            else
            {
                cerr<<"ERROR 06: Failed to acquire device number "<<deviceID<<endl;
                return -1;
            }
		}

		//  limit fps
		std::this_thread::sleep_until(loopStartTime + chrono::milliseconds((int)(1000/VJOYFPS)));
    }

    RelinquishVJD(deviceID);
    return 0;
}

int main(int argc, char *argv[])
{
    unsigned int deviceID = 1;
    if(argc > 1)
    {
        deviceID = atoi(argv[1]);
    }

    std::thread cvThread = std::thread(cvHandler);
    std::thread vJoyThread = std::thread(vJoyHandler, deviceID);

    char type;
    double newThreshold;

    while(!global_exit_flag)
    {
        cin>>type;

        if(type == 'q')
        {
            global_exit_flag = true;
        }
        else if(type == 's')
        {
            global_statistics_flag = !global_statistics_flag;
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

                case 'd':
                    mtx.lock();
                    global_deadzone = (uint32_t)newThreshold;
                    mtx.unlock();
            }
        }
    }
    cvThread.join();
    vJoyThread.join();
    cout<<"done";
    return 0;
}

