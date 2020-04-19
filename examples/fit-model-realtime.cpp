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
#include "linmath.h"

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
    GLuint vertex_buffer, vertex_shader, fragment_shader, program;
    GLint vpos_location, vcol_location;

    void framebuffer_size_callback(GLFWwindow *window, int width, int height);

    void processInput(GLFWwindow *window);

    const unsigned int SCR_WIDTH = 200;
    const unsigned int SCR_HEIGHT = 200;

    const char *vertexShaderSource = "#version 330 core\n"
                                     "layout (location = 0) in vec3 aPos;\n"
                                     "void main()\n"
                                     "{\n"
                                     "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
                                     "}\0";
    const char *fragmentShaderSource = "#version 330 core\n"
                                       "out vec4 FragColor;\n"
                                       "void main()\n"
                                       "{\n"
                                       "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
                                       "}\n\0";

    float vertices[] = {
            0.5f, 0.5f, 0.0f,  // top right
            0.5f, -0.5f, 0.0f,  // bottom right
            -0.5f, -0.5f, 0.0f,  // bottom left
            -0.5f, 0.5f, 0.0f   // top left
    };
    unsigned int indices[] = {  // note that we start from 0!
            0, 1, 3,  // first Triangle
            1, 2, 3   // second Triangle
    };
    unsigned int VBO, VAO, EBO;
    int shaderProgram;

    float faceVertices[3448];
    int faceTriangles[6736 * 3];

    bool displayFace = false;

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
        window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Preview", NULL, NULL);
        if (!window) {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return -1;
        }

        // Mathe the window's context current
        glfwMakeContextCurrent(window);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

        // Initialize the OpenGL API with GLAD
        if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
            std::cout << "Failed to initialize GLAD" << std::endl;
            return -1;
        }

        // build and compile our shader program
        // ------------------------------------
        // vertex shader
        int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(vertexShader);
        // check for shader compile errors
        int success;
        char infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        }
        // fragment shader
        int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);
        // check for shader compile errors
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        }
        // link shaders
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        // check for linking errors
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        // data
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
        glEnableVertexAttribArray(0);

        // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
        //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
        // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
        glBindVertexArray(0);


        // uncomment this call to draw in wireframe polygons.
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
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
            if (success && true) {
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
                        ibug_contour, model_contour, 20, cpp17::nullopt, 30.0f);

                /*
                // The 3D head pose can be recovered as follows:
                float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
                // and similarly for pitch and roll.

                // Extract the texture from the image using given mesh and camera parameters:
                const Eigen::Matrix<float, 3, 4> affine_from_ortho =
                        fitting::get_3x4_affine_camera_matrix(rendering_params, frame.cols, frame.rows);
                */

                auto finish = std::chrono::high_resolution_clock::now();

                // Draw the fitted mesh as wireframe, and save the image:
                /*
                render::draw_wireframe(outimg, mesh, rendering_params.get_modelview(),
                                       rendering_params.get_projection(),
                                       fitting::get_opencv_viewport(frame.cols, frame.rows));

                const core::Image4u isomap =
                        render::extract_texture(mesh, affine_from_ortho, core::from_mat(frame), true);
                        */

                std::chrono::duration<double> elapsed = finish - start;
                std::cout << "Elapsed time: " << elapsed.count() << " s\n";

                // display output
                // imshow("isomap", outimg); //core::to_mat(isomap));

                // store 3d model
                // core::write_textured_obj(mesh, "florian.obj");
                // cv::imwrite("florian.isomap.png", core::to_mat(isomap));

                // update vertices
                cout << "Verts: " << mesh.vertices.size() << " Tri: " << mesh.tvi.size() << endl;

                int i = 0;
                for (auto v : mesh.vertices) {
                    faceVertices[i++] = v.x() / 100.0f;
                    faceVertices[i++] = v.y() / 100.0f;
                    faceVertices[i++] = v.z() / 100.0f;
                }

                i = 0;
                for (auto v : mesh.tvi) {
                    faceTriangles[i++] = v[0];
                    faceTriangles[i++] = v[1];
                    faceTriangles[i++] = v[2];
                }

                cout << "x: " << faceVertices[0] << " y: " << faceVertices[1] << " z: " << faceVertices[2] << endl;
                cout << "t0: " << faceTriangles[0] << " t1: " << faceTriangles[1] << " t2: " << faceTriangles[2]
                     << endl;

                glBindBuffer(GL_ARRAY_BUFFER, VBO);
                glBufferData(GL_ARRAY_BUFFER, sizeof(faceVertices), faceVertices, GL_STATIC_DRAW);

                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(faceTriangles), faceTriangles, GL_STATIC_DRAW);

                displayFace = true;
            }

            // render 3d
            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // draw our first triangle
            glUseProgram(shaderProgram);
            glBindVertexArray(
                    VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized

            if (displayFace) {
                glPointSize(5.0f);
                glDrawElements(GL_POINTS, 6736 * 3, GL_UNSIGNED_INT, 0);
            } else {
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            }
            // glBindVertexArray(0); // no need to unbind it every time

            // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
            // -------------------------------------------------------------------------------
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
