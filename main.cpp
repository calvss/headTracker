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
int global_frame_height;
int global_frame_width;

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
//    cv::Mat img = cv::imread("arnold_schwarzenegger.jpg", cv::IMREAD_COLOR);
//    if(img.empty())
//       return -1;
//    cv::namedWindow("arnold_schwarzenegger", cv::WINDOW_AUTOSIZE );
//    cv::imshow("arnold_schwarzenegger", img);
//    cv::waitKey(0);
    cv::Point coords(0,0);

    vector<cv::Mat> hsvChannels;

    vector<cv::Mat> rawFrames;

    cv::Mat rgbFrame, hsvFrame, colorFrame;
    cv::Mat hues, saturations, values, newFrame, filteredFrame;

    int saturation;
    int lowerThreshold, upperThreshold;



    cv::VideoCapture camera(0);
    if(!camera.isOpened())
    {
        cerr << "ERROR: Could not open camera" << endl;
        return 1;
    }

    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 120);
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 160);

    global_frame_height = camera.get(cv::CAP_PROP_FRAME_HEIGHT);
    global_frame_width = camera.get(cv::CAP_PROP_FRAME_WIDTH);

    cout<<"resolution is: "<<global_frame_width<<" x "<<global_frame_height<<endl;

    cv::namedWindow("Hue", cv::WINDOW_NORMAL);
    cv::resizeWindow("Hue", 320, 240);
    cv::setMouseCallback("Hue", mouse_callback, &coords);

    cv::namedWindow("Filtered", cv::WINDOW_NORMAL);
    cv::resizeWindow("Filtered", 320, 240);

    cv::namedWindow("Color", cv::WINDOW_NORMAL);
    cv::resizeWindow("Color", 320, 240);

    cv::namedWindow("Process", cv::WINDOW_NORMAL);
    cv::resizeWindow("Process", 320, 240);

    cv::namedWindow("Keypoints", cv::WINDOW_NORMAL);
    cv::resizeWindow("Keypoints", 320, 240);

    auto startTime = chrono::system_clock::now();

    while (1)
    {

//          TODO: noise filtering? maybe
//        for(int i = 0; i < 3; i++)
//        {
//            camera>>rgbFrame;
//            rawFrames.push_back(rgbFrame);
//        }
        camera>>colorFrame;

        cv::Rect cropRect(0, global_frame_height/2, global_frame_width, global_frame_height/2);

        colorFrame = colorFrame(cropRect);

        cv::cvtColor(colorFrame, hsvFrame, cv::COLOR_BGR2HSV);
        cv::split(hsvFrame, hsvChannels);
        hues = hsvChannels[0];
        saturations = hsvChannels[1];
        values = hsvChannels[2];

        int frameVectorSize = (int) rawFrames.size();
        if(frameVectorSize >= NUM_NOISE_FRAMES)
        {
            rawFrames.pop_back();
        }
        rawFrames.emplace(rawFrames.begin(), hues.clone());

        if(frameVectorSize >= 3 && frameVectorSize%2 != 0)
        {
            cout<<"denoise "<<frameVectorSize<<" ";
            //  fastNlMeansDenoisingMulti( input array, output, index of image to filter, (n) images to process, filter strength, windowSize, searchWindowSize)
            //  NOTE: time complexity is O(searchWindowSize) + O(n)
            cv::fastNlMeansDenoisingMulti(rawFrames, filteredFrame, frameVectorSize/2, frameVectorSize, 20, 7, 11);
        }
        else
        {
            filteredFrame = rawFrames[0];
        }

        /* threshold types:
        0: binary
        1: binary inverted
        2: threshold truncate
        3: threshold to zero
        4: threshold to zero inverted
        */

        mtx.lock();
        lowerThreshold = global_threshold_lower;
        upperThreshold = global_threshold_upper;
        mtx.unlock();

        cv::threshold(filteredFrame, newFrame, lowerThreshold, 255, 3);
        cv::threshold(newFrame, newFrame, upperThreshold, 255, 4); //obtain only pixels in between the bounds, and change to binary values
        cv::threshold(newFrame, newFrame, 1, 255, 0);

        cv::circle(hues, coords, 5, cv::Scalar(255, 0, 0));

        cout<<coords.x<<", "<<coords.y<<": ";
        saturation = (int)hues.at<uchar>(coords);
        cout<<saturation<<", ";

        cv::imshow("Hue", hues);
        cv::imshow("Filtered", filteredFrame);
        cv::imshow("Color", colorFrame);
        cv::imshow("Process", newFrame);


        // wait (5ms) for a key to be pressed
        if (cv::waitKey(5) >= 0)
        {
            break;
        }

        cv::SimpleBlobDetector::Params params;
        params.filterByArea = true;

        mtx.lock();
        params.minArea = global_blob_size;
        mtx.unlock();

        params.filterByInertia = false;
        params.filterByConvexity = false;
        params.filterByCircularity = false;
        params.filterByColor = false;

        vector<cv::KeyPoint> keypoints;
        cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
        detector->detect(newFrame, keypoints);

        cv::Mat frameWithKeyPoints;
        cv::drawKeypoints(newFrame, keypoints, frameWithKeyPoints, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::imshow("Keypoints", frameWithKeyPoints);


        cout<<upperThreshold<<"u "<<lowerThreshold<<"l ";

        auto timeNow = chrono::system_clock::now();
        auto frameDuration = chrono::duration_cast<chrono::milliseconds>(timeNow - startTime);
        int frameMs = frameDuration.count();
        startTime = timeNow;
        cout<<(1000./frameMs)<<" FPS"<<endl;

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

