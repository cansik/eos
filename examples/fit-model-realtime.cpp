#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/face.hpp>

#include <iostream>
#include <cstdio>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace cv::face;
using namespace std;

//hide the local functions in an anon namespace
namespace {
    void help(char **av) {
        cout
                << "The program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer."
                << endl
                << "Usage:\n" << av[0] << " <video file, image sequence or device number>" << endl
                << "q,Q,esc -- quit" << endl
                << "space   -- save frame" << endl << endl
                << "\tTo capture from a camera pass the device number. To find the device number, try ls /dev/video*"
                << endl
                << "\texample: " << av[0] << " 0" << endl
                << "\tYou may also pass a video file instead of a device number" << endl
                << "\texample: " << av[0] << " video.avi" << endl
                << "\tYou can also pass the path to an image sequence and OpenCV will treat the sequence just like a video."
                << endl
                << "\texample: " << av[0] << " right%%02d.jpg" << endl;
    }

    int process(VideoCapture &capture) {
        int n = 0;
        char filename[200];
        string window_name = "video | q or esc to quit";
        cout << "press space to save a picture. q or esc to quit" << endl;
        namedWindow(window_name, WINDOW_KEEPRATIO); //resizable window;
        Mat frame, gray;

        // face detection and landmark extraction
        CascadeClassifier faceDetector("data/haarcascade_frontalface_alt2.xml");
        Ptr<Facemark> facemark = FacemarkLBF::create();
        facemark->loadModel("data/lbfmodel.yaml");

        for (;;) {
            capture >> frame;
            if (frame.empty())
                break;

            // detect single face
            vector<Rect> faces;
            vector<vector<Point2f> > landmarks;

            cvtColor(frame, gray, COLOR_BGR2GRAY);
            faceDetector.detectMultiScale(gray, faces);

            // extract landmarks
            bool success = facemark->fit(frame, faces, landmarks);

            if (success) {
                // If successful, render the landmarks on the face
                for (int i = 0; i < landmarks.size(); i++) {
                    drawFacemarks(frame, landmarks[i], Scalar(0, 0, 255));
                }
            }

            // display frame
            imshow(window_name, frame);
            char key = (char) waitKey(30); //delay N millis, usually long enough to display and capture input

            switch (key) {
                case 'q':
                case 'Q':
                case 27: //escape key
                    return 0;
                case ' ': //Save an image
                    sprintf(filename, "filename%.3d.jpg", n++);
                    imwrite(filename, frame);
                    cout << "Saved " << filename << endl;
                    break;
                default:
                    break;
            }
        }
        return 0;
    }
}

int main(int ac, char **av) {
    cv::CommandLineParser parser(ac, av, "{help h||}{@input||}");
    if (parser.has("help")) {
        help(av);
        return 0;
    }
    std::string arg = parser.get<std::string>("@input");
    if (arg.empty()) {
        help(av);
        return 1;
    }
    VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file or image sequence
    if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
        capture.open(atoi(arg.c_str()));
    if (!capture.isOpened()) {
        cerr << "Failed to open the video device, video file or image sequence!\n" << endl;
        help(av);
        return 1;
    }
    return process(capture);
}
