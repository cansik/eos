#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/face.hpp>

#include <iostream>
#include <cstdio>
#include <opencv2/imgproc.hpp>

#include "eos/core/Image.hpp"
#include "eos/core/image/opencv_interop.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/core/write_obj.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/render/texture_extraction.hpp"

#include "Eigen/Core"

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include <iostream>
#include <vector>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using Eigen::Vector2f;
using Eigen::Vector4f;
using std::cout;
using std::endl;
using std::string;
using std::vector;

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
        namedWindow("isomap", WINDOW_KEEPRATIO); //resizable window;
        Mat frame, gray;

        // face detection and landmark extraction
        CascadeClassifier faceDetector("data/haarcascade_frontalface_alt2.xml");
        Ptr<Facemark> facemark = FacemarkLBF::create();
        facemark->loadModel("data/lbfmodel.yaml");

        // face fitting stuff
        cout << "loading morphable model..." << endl;
        morphablemodel::MorphableModel morphable_model;
        try {
            morphable_model = morphablemodel::load_model("../share/sfm_shape_3448.bin");
        } catch (const std::runtime_error &e) {
            cout << "Error loading the Morphable Model: " << e.what() << endl;
            return EXIT_FAILURE;
        }

        cout << "loading landmark mapper..." << endl;
        core::LandmarkMapper landmark_mapper;
        try {
            landmark_mapper = core::LandmarkMapper("../share/ibug_to_sfm.txt");
        } catch (const std::exception &e) {
            cout << "Error loading the landmark mappings: " << e.what() << endl;
            return EXIT_FAILURE;
        }

        for (;;) {
            capture >> frame;
            if (frame.empty())
                break;

            // detect single face
            vector<Rect> faces;
            vector<vector<Point2f> > landmarksByFace;

            cvtColor(frame, gray, COLOR_BGR2GRAY);
            faceDetector.detectMultiScale(gray, faces);

            // extract landmarks
            bool success = facemark->fit(frame, faces, landmarksByFace);

            if (success) {
                // If successful, render the landmarks on the face
                for (int i = 0; i < landmarksByFace.size(); i++) {
                    drawFacemarks(frame, landmarksByFace[i], Scalar(0, 0, 255));
                }
            }

            // start fitting process
            // todo: do fitting of face model here
            if (success) {
                // copy landmark to eigen model
                LandmarkCollection<Eigen::Vector2f> landmarks;
                for (int i = 0; i < landmarksByFace[0].size(); i++) {
                    Landmark<Vector2f> landmark;
                    landmark.name = std::to_string(i + 1);
                    landmark.coordinates[0] = landmarksByFace[0][i].x;
                    landmark.coordinates[1] = landmarksByFace[0][i].y;
                    landmarks.emplace_back(landmark);
                }

                // start fitting
                // These will be the final 2D and 3D points used for the fitting:
                vector<Vector4f> model_points; // the points in the 3D shape model
                vector<int> vertex_indices;    // their vertex indices
                vector<Vector2f> image_points; // the corresponding 2D landmark points

                // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
                for (int i = 0; i < landmarks.size(); ++i) {
                    auto converted_name = landmark_mapper.convert(landmarks[i].name);
                    if (!converted_name) { // no mapping defined for the current landmark
                        continue;
                    }
                    int vertex_idx = std::stoi(converted_name.value());
                    auto vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
                    model_points.emplace_back(Vector4f(vertex.x(), vertex.y(), vertex.z(), 1.0f));
                    vertex_indices.emplace_back(vertex_idx);
                    image_points.emplace_back(landmarks[i].coordinates);
                }

                // Estimate the camera (pose) from the 2D - 3D point correspondences
                fitting::ScaledOrthoProjectionParameters pose =
                        fitting::estimate_orthographic_projection_linear(image_points, model_points, true, frame.rows);
                fitting::RenderingParameters rendering_params(pose, frame.cols, frame.rows);

                // The 3D head pose can be recovered as follows:
                const float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
                // and similarly for pitch and roll.

                // Estimate the shape coefficients by fitting the shape to the landmarks:
                const Eigen::Matrix<float, 3, 4> affine_from_ortho =
                        fitting::get_3x4_affine_camera_matrix(rendering_params, frame.cols, frame.rows);
                const vector<float> fitted_coeffs = fitting::fit_shape_to_landmarks_linear(
                        morphable_model.get_shape_model(), affine_from_ortho, image_points, vertex_indices);

                // Obtain the full mesh with the estimated coefficients:
                const core::Mesh mesh = morphable_model.draw_sample(fitted_coeffs, vector<float>());

                // Extract the texture from the image using given mesh and camera parameters:
                const core::Image4u isomap = render::extract_texture(mesh, affine_from_ortho, core::from_mat(frame));

                // display isomap
                imshow("isomap", core::to_mat(isomap));
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

        return EXIT_SUCCESS;
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
