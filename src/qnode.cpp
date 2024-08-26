/**
 * @file /src/qnode.cpp
 *
 * @brief Ros communication central!
 *
 * @date February 2011
 **/

/*****************************************************************************
** Includes
*****************************************************************************/

#include <ros/ros.h>
#include <ros/network.h>
#include <string>
#include <std_msgs/String.h>
#include <sstream>
#include "../include/defense_vision_24/qnode.hpp"

#include <ros/package.h>
#include <fstream>
/*****************************************************************************
** Namespaces
*****************************************************************************/

namespace defense_vision_24
{
  /*****************************************************************************
  ** Implementation
  *****************************************************************************/

  QNode::QNode(int argc, char **argv) : init_argc(argc), init_argv(argv)
  {
    std::string packagePath = ros::package::getPath("defense_vision_24"); // ROS 패키지에서 경로 가져오기
    std::cout << packagePath << std::endl;                                // 패키지 경로 출력
    std::string dir = packagePath + "/yolo/";                             // YOLO 디렉터리 경로 설정
    {
      std::ifstream class_file(dir + "classes.txt"); // 클래스 파일 열기
      if (!class_file)
      {
        std::cerr << "failed to open .txt\n";
      }

      std::string line; // 클래스 이름을 읽어와 벡터에 추가
      while (std::getline(class_file, line))
        class_names.push_back(line);
    }

    std::string modelConfiguration = dir + "yolov7_tiny_defense2.cfg";     // 모델 구성 파일 경로 설정
    std::string modelWeights = dir + "yolov7_tiny_defense2_final.weights"; // 모델 가중치 파일 경로 설정

    net = cv::dnn::readNetFromDarknet(modelConfiguration,
                                      modelWeights); // Darknet 형식의 모델 파일로부터 네트워크 구성 및 가중치 로드
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA); // CUDA 백엔드 및 타겟 설정 (주석 처리됨)
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); // OpenCV 백엔드와 CPU 타겟으로 설정
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }

  QNode::~QNode()
  {
    if (ros::isStarted())
    {
      ros::shutdown(); // explicitly needed since we use ros::start();
      ros::waitForShutdown();
    }
    wait();
  }

  bool QNode::init()
  {
    ros::init(init_argc, init_argv, "defense_vision_24");
    if (!ros::master::check())
    {
      return false;
    }
    ros::start(); // explicitly needed since our nodehandle is going out of scope.
    ros::NodeHandle n;

    // Add your ros communications here.
    n.param<std::string>("cam1_topic", cam1_topic_name, "/camera/color/image_raw");
    n.param<std::string>("cam2_topic", cam2_topic_name, "/usb_cam/image_raw");
    ROS_INFO("Starting Rescue Vision With Camera : %s", cam1_topic_name.c_str());

    // <-> cam*_topic_name
    image_transport::ImageTransport image(n);                                    // 이미지 전송을 위한 ImageTransport 객체 생성
    subImage = image.subscribe(cam1_topic_name, 1, &QNode::callbackImage, this); // 서브스크라이버

    // <-> depth cam
    image_transport::ImageTransport it_(n);
    std::string image_depth = n.resolveName("camera/aligned_depth_to_color/image_raw");
    sub_ = it_.subscribeCamera(image_depth, 1024, &QNode::callbackDepth, this);

    // <-> mani
    mani_vision_pub = n.advertise<mobile_base_msgs::mani_vision_defense>("/mvis_val", 1);
    mani_vision_sub = n.subscribe("/mvis_mode", 1, &QNode::mvis_callback, this);

    start();
    return true;
  }

  void QNode::run()
  {
    ros::Rate loop_rate(33);
    while (ros::ok())
    {
      ros::spinOnce();
      loop_rate.sleep();
    }
    std::cout << "Ros shutdown, proceeding to close the gui." << std::endl;
    Q_EMIT rosShutdown(); // used to signal the gui for a shutdown (useful to roslaunch)
  }

  // <-> cam*_topic_name
  void QNode::callbackImage(const sensor_msgs::ImageConstPtr &msg_img)
  {
    if (imgRaw == NULL && !isreceived) // imgRaw -> NULL, isreceived -> false
    {
      // ROS 이미지 메시지를 OpenCV Mat 형식으로 변환, 이미지 객체에 할당
      imgRaw = new cv::Mat(cv_bridge::toCvCopy(msg_img, sensor_msgs::image_encodings::RGB8)->image);

      if (imgRaw != NULL) // imgRaw 변환 성공
      {
        Q_EMIT sigRcvImg(); // 이미지 수신을 알리는 시그널 발생
        isreceived = true;
      }
    }
  }

  // <-> depth cam
  void QNode::callbackDepth(const sensor_msgs::ImageConstPtr &image_msg, const sensor_msgs::CameraInfoConstPtr &info_msg)
  {
    cv::Mat image;
    cv_bridge::CvImagePtr input_bridge;
    try
    {
      input_bridge = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::TYPE_16UC1);
      image = input_bridge->image;
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("Failed to convert depth image");
      return;
    }

    // 이미지의 중심 좌표 계산
    int center_x = 320;
    int center_y = 180 + 90;

    depth = image.at<short int>(cv::Point(center_x, center_y));

    // std::cout << "dp  "<< depth << std::endl;
  }

  //<-> mani
  void QNode::mvis_callback(const mobile_base_msgs::mani_vision_defense &msg)
  {
    mvis_sub = msg;
  }

} // namespace defense_vision_24
