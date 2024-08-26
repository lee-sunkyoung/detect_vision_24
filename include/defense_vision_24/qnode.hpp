/**
 * @file /include/defense_vision_24/qnode.hpp
 *
 * @brief Communications central!
 *
 * @date February 2011
 **/
/*****************************************************************************
** Ifdefs
*****************************************************************************/

#ifndef defense_vision_24_QNODE_HPP_
#define defense_vision_24_QNODE_HPP_

/*****************************************************************************
** Includes
*****************************************************************************/

// To workaround boost/qt4 problems that won't be bugfixed. Refer to
//    https://bugreports.qt.io/browse/QTBUG-22829
#ifndef Q_MOC_RUN
#include <ros/ros.h>
#endif
#include <string>
#include <QThread>
#include <QStringListModel>

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <opencv2/dnn.hpp>             // OpenCV 딥 러닝 관련 헤더 파일
#include <opencv2/dnn/all_layers.hpp>  // OpenCV 딥 러닝 모든 레이어 헤더 파일
#include "librealsense2/rsutil.h"
#include <mobile_base_msgs/mani_vision_defense.h>
/*****************************************************************************
** Namespaces
*****************************************************************************/

namespace defense_vision_24
{
/*****************************************************************************
** Class
*****************************************************************************/

class QNode : public QThread
{
  Q_OBJECT
public:
  QNode(int argc, char** argv);
  virtual ~QNode();
  bool init();
  void run();

  cv::Mat* imgRaw = NULL;                // 원본 이미지를 가리키는 포인터
  bool isreceived = false;               // 수신 여부를 나타내는 플래그
  cv::dnn::Net net;                      // DNN 네트워크
  std::vector<std::string> class_names;  // 클래스 이름 벡터
  float depth;

  std::string cam1_topic_name;
  std::string cam2_topic_name;

  // <-> mani
  ros::Publisher mani_vision_pub;
  ros::Subscriber mani_vision_sub;
  void mvis_callback(const mobile_base_msgs::mani_vision_defense& msg);

  mobile_base_msgs::mani_vision_defense mvis_pub;
  mobile_base_msgs::mani_vision_defense mvis_sub;
Q_SIGNALS:
  void rosShutdown();
  void sigRcvImg();

private:
  int init_argc;
  char** init_argv;

  // <-> cam*_topic_name
  image_transport::Subscriber subImage;                           // 서브스크라이버
  void callbackImage(const sensor_msgs::ImageConstPtr& msg_img);  // 이미지 콜백 함수 선언

  // <-> depth cam
  image_transport::CameraSubscriber sub_;
  void callbackDepth(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg);
};

}  // namespace defense_vision_24

#endif /* defense_vision_24_QNODE_HPP_ */
