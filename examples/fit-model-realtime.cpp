#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/face.hpp>

#include "eos/core/Image.hpp"
#include "eos/core/image/opencv_interop.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/core/write_obj.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/render/draw_utils.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/cpp17/optional.hpp"

#include "Eigen/Core"

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <chrono>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using Eigen::Vector2f;
using Eigen::Vector4f;
using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;

using namespace cv;
using namespace cv::face;
using namespace std;

//hide the local functions in an anon namespace
namespace {
    GLFWwindow *window;

    //void framebuffer_size_callback(GLFWwindow *window, int width, int height);

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

    int setup3dPreview() {
        // Initialize the library
        if (!glfwInit())
            return -1;

        // Define version and compatibility settings
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

        // Create a windowed mode window and its OpenGL context
        window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
        if (!window) {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return -1;
        }

        // Mathe the window's context current
        glfwMakeContextCurrent(window);
        //glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

        // Initialize the OpenGL API with GLAD
        if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
            std::cout << "Failed to initialize GLAD" << std::endl;
            return -1;
        }
    }

    void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
        // make sure the viewport matches the new window dimensions; note that width
        // and height will be significantly larger than specified on retina displays
        glViewport(0, 0, width, height);
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
        CascadeClassifier faceDetector("data/haarcascade_frontalface_default.xml");
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

        // These two are used to fit the front-facing contour to the ibug contour landmarks:
        const fitting::ModelContour model_contour = fitting::ModelContour::load("../share/sfm_model_contours.json");
        const fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load("../share/ibug_to_sfm.txt");

        // The edge topology is used to speed up computation of the occluding face contour fitting:
        const morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(
                "../share/sfm_3448_edge_topology.json");

        const vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(
                "../share/expression_blendshapes_3448.bin");

        morphablemodel::MorphableModel morphable_model_with_expressions(
                morphable_model.get_shape_model(), blendshapes, morphable_model.get_color_model(), cpp17::nullopt,
                morphable_model.get_texture_coordinates());

        setup3dPreview();

        for (;;) {
            capture >> frame;
            if (frame.empty())
                break;

            Mat outimg = frame.clone();

            // detect single face
            vector<Rect> faces;
            vector<vector<Point2f> > landmarksByFace;

            cvtColor(frame, gray, COLOR_BGR2GRAY);
            faceDetector.detectMultiScale(gray, faces);

            // extract landmarks
            bool success = facemark->fit(frame, faces, landmarksByFace);

            if (success) {
                // If successful, render the landmarks on the face
                drawFacemarks(frame, landmarksByFace[0], Scalar(0, 0, 255));
            }

            // start fitting process
            // todo: do fitting of face model here
            if (success && false) {
                auto start = std::chrono::high_resolution_clock::now();

                // copy landmark to eigen model
                LandmarkCollection<Eigen::Vector2f> landmarks;
                for (int i = 0; i < landmarksByFace[0].size(); i++) {
                    Landmark<Vector2f> landmark;
                    landmark.name = std::to_string(i + 1);
                    landmark.coordinates[0] = landmarksByFace[0][i].x;
                    landmark.coordinates[1] = landmarksByFace[0][i].y;
                    landmarks.emplace_back(landmark);
                }

                // Fit the model, get back a mesh and the pose:
                core::Mesh mesh;
                fitting::RenderingParameters rendering_params;
                std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(
                        morphable_model_with_expressions, landmarks, landmark_mapper, frame.cols, frame.rows,
                        edge_topology,
                        ibug_contour, model_contour, 5, cpp17::nullopt, 30.0f);

                // The 3D head pose can be recovered as follows:
                float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
                // and similarly for pitch and roll.

                // Extract the texture from the image using given mesh and camera parameters:
                const Eigen::Matrix<float, 3, 4> affine_from_ortho =
                        fitting::get_3x4_affine_camera_matrix(rendering_params, frame.cols, frame.rows);

                auto finish = std::chrono::high_resolution_clock::now();

                // Draw the fitted mesh as wireframe, and save the image:
                render::draw_wireframe(outimg, mesh, rendering_params.get_modelview(),
                                       rendering_params.get_projection(),
                                       fitting::get_opencv_viewport(frame.cols, frame.rows));

                const core::Image4u isomap =
                        render::extract_texture(mesh, affine_from_ortho, core::from_mat(frame), true);

                std::chrono::duration<double> elapsed = finish - start;
                std::cout << "Elapsed time: " << elapsed.count() << " s\n";

                // display output
                imshow("isomap", outimg); //core::to_mat(isomap));

                // store 3d model
                core::write_textured_obj(mesh, "florian.obj");
                cv::imwrite("florian.isomap.png", core::to_mat(isomap));
            }

            // render 3d
            glClear(GL_COLOR_BUFFER_BIT);
            glfwSwapBuffers(window);
            glfwPollEvents();

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

        glfwDestroyWindow(window);
        glfwTerminate();
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
