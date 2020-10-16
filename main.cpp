#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<future>
#include<mutex>
#include<chrono>
#include<thread>

#define SPLIT_COLS 3
#define SPLIT_ROWS 2
#define NUM_NOISE_FRAMES 2

using namespace std;

mutex mtx;
volatile int global_threshold_upper = 45;
volatile int global_threshold_lower = 33;
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

    cv::Mat rawFrames[NUM_NOISE_FRAMES];

    cv::Mat rgbFrame, hsvFrame, colorFrame;
    cv::Mat hues, saturations, values, newFrame, lastFrame, unfilteredFrame;

    int saturation;
    int lowerThreshold, upperThreshold;



    cv::VideoCapture camera(0);
    if(!camera.isOpened())
    {
        cerr << "ERROR: Could not open camera" << endl;
        return 1;
    }

    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 320);

    global_frame_height = camera.get(cv::CAP_PROP_FRAME_HEIGHT);
    global_frame_width = camera.get(cv::CAP_PROP_FRAME_WIDTH);

    cout<<"resolution is: "<<global_frame_width<<" x "<<global_frame_height<<endl;

    cv::namedWindow("Hue", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Hue", mouse_callback, &coords);
    cv::namedWindow("Saturation", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Color", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Process", cv::WINDOW_AUTOSIZE);

    auto startTime = chrono::system_clock::now();

    while (1)
    {

//          TODO: noise filtering? maybe
//        for(int i = 0; i < 3; i++)
//        {
//            camera>>rgbFrame;
//            rawFrames.push_back(rgbFrame);
//        }

        for(int i = 0; i < NUM_NOISE_FRAMES; i++)
        {
            camera>>colorFrame;

            cv::Rect cropRect(0, 120, 320, 120);

            colorFrame = colorFrame(cropRect);

            cout<<"cropped";


            rawFrames[i] = colorFrame.clone();


            cv::cvtColor(rawFrames[i], hsvFrame, cv::COLOR_BGR2HSV);
            cv::split(hsvFrame, hsvChannels);
            hues = hsvChannels[0];
            saturations = hsvChannels[1];
            values = hsvChannels[2];

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

            //cv::fastNlMeansDenoising(hues, hues, 15, 3, 5);

            cv::threshold(hues, rawFrames[i], lowerThreshold, 255, 3);
            cv::threshold(rawFrames[i], rawFrames[i], upperThreshold, 255, 4); //obtain only pixels in between the bounds, and change to binary values
            cv::threshold(rawFrames[i], rawFrames[i], 1, 255, 0);
        }

        if(NUM_NOISE_FRAMES > 1)
        {
            cv::bitwise_and(rawFrames[0], rawFrames[1], newFrame);
            for(int i = 1; i < NUM_NOISE_FRAMES - 1; i++)
            {
                cv::bitwise_and(newFrame, rawFrames[i], newFrame);
            }
        }
        else
        {
            newFrame = rawFrames[0];
        }

        //cv::fastNlMeansDenoisingColoredMulti(rgbFrame, rgbFrame, 4, 4, 5, 15);

//        camera>>rgbFrame;
//        cv::cvtColor(rgbFrame, hsvFrame, cv::COLOR_BGR2HSV);
//        cv::split(hsvFrame, hsvChannels);
//        hues = hsvChannels[0];
//        saturations = hsvChannels[1];
//        values = hsvChannels[2];
//
//        /* threshold types:
//        0: binary
//        1: binary inverted
//        2: threshold truncate
//        3: threshold to zero
//        4: threshold to zero inverted
//        */
//
//        mtx.lock();
//        lowerThreshold = global_threshold_lower;
//        upperThreshold = global_threshold_upper;
//        mtx.unlock();
//
//        cv::threshold(hues, newFrame, lowerThreshold, 255, 3);
//        cv::threshold(newFrame, newFrame, upperThreshold, 255, 4); //obtain only pixels in between the bounds, and change to binary values
//        cv::threshold(newFrame, newFrame, 1, 255, 0);

        cv::circle(hues, coords, 5, cv::Scalar(255, 0, 0));

        cout<<coords.x<<", "<<coords.y<<": ";
        saturation = (int)hues.at<uchar>(coords);
        cout<<saturation<<", ";

        cv::imshow("Hue", hues);
        cv::imshow("Saturation", saturations);
        //if(!lastFrame.empty())
        cv::imshow("Color", colorFrame);
        cv::imshow("Process", newFrame);

        lastFrame = newFrame.clone();

        // wait (5ms) for a key to be pressed
        if (cv::waitKey(5) >= 0)
        {
            break;
        }

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
    }
    cout<<"done";
    return 0;
}

