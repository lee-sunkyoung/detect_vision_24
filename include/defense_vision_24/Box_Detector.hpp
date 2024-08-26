/*****************************************************************************
** Ifdefs
*****************************************************************************/

#ifndef defense_vision_24_BOX_DETECTOR_HPP_
#define defense_vision_24_BOX_DETECTOR_HPP_

/*****************************************************************************
** Includes
*****************************************************************************/

#ifndef Q_MOC_RUN
#include <ros/ros.h>
#endif
#include <string>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
/*****************************************************************************
** Namespaces
*****************************************************************************/

namespace defense_vision_24
{
/*****************************************************************************
** Class
*****************************************************************************/

class Box_Detector
{
public:
  cv::Mat Binary(cv::Mat& img, int val[]);
  cv::Mat canny_edge(cv::Mat& binary_img);
  cv::Mat region_of_interest(cv::Mat& edge_img, int val[]);
  std::vector<cv::Vec4i> houghlines(cv::Mat roi_img);
  bool checkLineIntersection(cv::Vec4i l1, cv::Vec4i l2, cv::Point2f& intersection);
  float distance(const cv::Point2f& p1, const cv::Point2f& p2);
  cv::Point2f findCenter(const cv::Rect& rect);

private:
};

}  // namespace defense_vision_24

#endif /* defense_vision_24_BOX_DETECTOR_HPP_ */
