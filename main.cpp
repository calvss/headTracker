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


#define NUM_NOISE_FRAMES 3 // must be odd

using namespace std;

mutex mtx;
volatile int global_deadzone = 100;                 //  centidegrees
volatile int global_vjoy_fps = 60;
volatile int global_cv_fps = 24;
volatile int global_threshold_upper = 170;
volatile int global_threshold_lower = 130;
volatile int global_angle_offset = -500;               //  centidegrees
volatile double global_filter_constant = 0.3;
volatile double global_head_angle = M_PI/2;         //  radians
volatile double global_angle_sensitivity = 0.2;
volatile double global_kp = 0.9;
volatile double global_ki = 0.1;
volatile double global_kd = 2.0;
volatile bool global_exit_flag = false;
volatile bool global_statistics_flag = false;
const int global_const_frame_height = 240;
const int global_const_frame_width = 320;

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

    camera.set(cv::CAP_PROP_FRAME_WIDTH, global_const_frame_width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, global_const_frame_height);

    cout<<"resolution is: "<<global_const_frame_width<<" x "<<global_const_frame_height<<endl;


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

    bool exitLoop = false;
    while(!exitLoop)
    {
        auto loopStartTime = chrono::system_clock::now();
        mtx.lock();
        exitLoop = global_exit_flag;
        mtx.unlock();

        //  capture an image and crop
        cv::Mat colorFrame;
        camera>>colorFrame;
        cv::Rect cropRect(0, global_const_frame_height/2, global_const_frame_width, global_const_frame_height/2);
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
            cv::fastNlMeansDenoisingMulti(rawFrames, filteredFrame, frameVectorSize/2, frameVectorSize, 20, 5, 7);
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

        //  morphology based filtering
        cv::Mat se1 = getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::Mat se2 = getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(5, 5));
        // cv::Mat se3 = getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(1, 1));
        cv::Mat cleanFrame;
        cv::morphologyEx(newFrame, cleanFrame, cv::MorphTypes::MORPH_OPEN, se2); //  opening eliminates small dots
        cv::morphologyEx(cleanFrame, cleanFrame, cv::MorphTypes::MORPH_CLOSE, se1);  //  closing eliminates small holes

        //  detect the contours
        vector<vector<cv::Point>> contours;
        cv::findContours(cleanFrame, contours, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_NONE);

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
            cerr<<"WARN: no contours"<<endl;
            continue;
        }

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

        mtx.lock();
        double filterConstant = global_filter_constant;
        mtx.unlock();
        //  low pass filter
        newAngle = ((1. - filterConstant) * previousAngle) + (filterConstant * (newAngle + previousAngle)/2);
        previousAngle = newAngle;

        mtx.lock();
        global_head_angle = newAngle;
        mtx.unlock();

        //  generate a color image to draw the contour outlines
        cv::Mat frameWithContours(cleanFrame.size(), CV_8UC3); //  uint8, 3 channels
        cv::cvtColor(cleanFrame, frameWithContours, cv::COLOR_GRAY2BGR);
        cv::drawContours(frameWithContours, biggestContours, -1, cv::Scalar(255, 0, 0));

        //  draw the line connecting the centroids
        cv::line(frameWithContours, centroid1, centroid2, cv::Scalar(255, 255, 255));

        cv::line(frameWithContours, cv::Point(50,0), cv::Point(50*cos(newAngle) + 50, 50*sin(newAngle)), cv::Scalar(255, 255, 255));

        //  draw a circle at mouse cursor
        cv::circle(hueFrame, mouseCoords, 5, cv::Scalar(255, 0, 0));

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

        mtx.lock();
        bool printStatistics = global_statistics_flag;
        mtx.unlock();

        if(printStatistics)
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
        if (cv::waitKey(1) >= 0)
        {
            mtx.lock();
            global_exit_flag = true;
            mtx.unlock();
        }

        //  limit fps
        mtx.lock();
		int cvFPS = global_cv_fps;
		mtx.unlock();
		std::this_thread::sleep_until(loopStartTime + chrono::milliseconds((int)(1000/cvFPS)));
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
    int32_t previousHeadAngle = 9000;
    int32_t headAngleSetpoint = 9000;
    int64_t errorIntegral = 0;
    int32_t previousError = 0;
    bool exitLoop = false;
    while(!exitLoop)
    {
        auto loopStartTime = chrono::system_clock::now();

        mtx.lock();
        exitLoop = global_exit_flag;
        double headAngle = global_head_angle;
        double filterConstant = global_filter_constant;
        double angleSensitivity = global_angle_sensitivity;
        double kP = global_kp;
        double kI = global_ki;
        double kD = global_kd;
        int vjoyFPS = global_vjoy_fps;
        int deadzone = global_deadzone;
        int angleOffset = global_angle_offset;
        mtx.unlock();

        //  POV hat position is an integer from 0 to 35999 (inclusive), set the value to 0xFFFFFFF for neutral
        //  represents the angle in centidegrees, with 0 at straight forward
        //  headAngle is in radians with 0 at pure left
        //  headAngleSetpoint, previousHeadAngle is in centidegrees with 0 at pure left

        headAngleSetpoint = (int32_t)(headAngle * 18000./M_PI);
        headAngleSetpoint = headAngleSetpoint + angleOffset;

        //  gotta rotate the hand angle before changing sensitivity
        int32_t headAngleFromForward = headAngleSetpoint - 9000;
        headAngleFromForward = (int32_t)(headAngleFromForward * angleSensitivity);
        headAngleSetpoint = headAngleFromForward + 9000;

        int32_t error = headAngleSetpoint - previousHeadAngle;
        int32_t errorDerivative = error - previousError;
        errorIntegral += error;


        headAngleSetpoint = (int32_t)(error * kP) + (int32_t)(errorIntegral * kI) + (int32_t)(errorDerivative * kD);

        //  low-pass filtering
        previousHeadAngle = (int32_t)(((1. - filterConstant) * (double)previousHeadAngle) + (filterConstant * (double)(headAngleSetpoint + previousHeadAngle)/2.));

        previousError = error;

        //  convert headAngle into centidegrees and rotate by 90 degrees
        uint32_t hatPos = (35900 - (((uint32_t)previousHeadAngle) + 9000)) % 35900;
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
		std::this_thread::sleep_until(loopStartTime + chrono::milliseconds((int)(1000/vjoyFPS)));
    }

    RelinquishVJD(deviceID);
    std::terminate();
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

    bool exitLoop = false;
    while(!exitLoop)
    {
        mtx.lock();
        exitLoop = global_exit_flag;
        mtx.unlock();

        cin>>type;

        if(type == 'q')
        {
            mtx.lock();
            global_exit_flag = true;
            mtx.unlock();
        }
        else if(type == 's')
        {
            mtx.lock();
            global_statistics_flag = !global_statistics_flag;
            mtx.unlock();
        }
        else if(type == 'h')
        {
            cout<<"Help:"<<endl;
            cout<<"Command syntax: [acfhloqsuvzpid] [<value>]"<<endl;
            cout<<"h: help"<<endl;
            cout<<"q: quit"<<endl;
            cout<<"s: toggle statistics"<<endl;
            cout<<"u    [int]: upper threshold"<<endl;
            cout<<"l    [int]: lower threshold"<<endl;
            cout<<"a [double]: low-pass filter constant"<<endl;
            cout<<"z    [int]: deadzone in centidegrees"<<endl;
            cout<<"f    [int]: vJoy controller update rate"<<endl;
            cout<<"v    [int]: sensitivity"<<endl;
            cout<<"o    [int]: offset"<<endl;
            cout<<"c    [int]: computer vision framerate"<<endl;
            cout<<"p [double]: PID controller kP"<<endl;
            cout<<"i [double]: PID controller kI"<<endl;
            cout<<"d [double]: PID controller kD"<<endl;
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
                    break;

                case 'l':
                    mtx.lock();
                    global_threshold_lower = (int) newThreshold;
                    mtx.unlock();
                    break;

                case 'a':
                    mtx.lock();
                    global_filter_constant = newThreshold;
                    mtx.unlock();
                    break;

                case 'z':
                    mtx.lock();
                    global_deadzone = (int) newThreshold;
                    mtx.unlock();
                    break;

                case 'f':
                    mtx.lock();
                    global_vjoy_fps = (int) newThreshold;
                    mtx.unlock();
                    break;

                case 'v':
                    mtx.lock();
                    global_angle_sensitivity = newThreshold;
                    mtx.unlock();
                    break;

                case 'o':
                    mtx.lock();
                    global_angle_offset = (int) newThreshold;
                    mtx.unlock();
                    break;

                case 'c':
                    mtx.lock();
                    global_cv_fps = (int) newThreshold;
                    mtx.unlock();
                    break;

                case 'p':
                    mtx.lock();
                    global_kp = newThreshold;
                    mtx.unlock();
                    break;

                case 'i':
                    mtx.lock();
                    global_ki = newThreshold;
                    mtx.unlock();
                    break;

                case 'd':
                    mtx.lock();
                    global_kd = newThreshold;
                    mtx.unlock();
                    break;
            }
        }
    }
    cvThread.join();
    vJoyThread.join();
    cout<<"done";
    return 0;
}

