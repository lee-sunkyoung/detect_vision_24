/*****************************************************************************
** Includes
*****************************************************************************/
#include <ros/ros.h>
#include <ros/network.h>
#include <string>
#include <std_msgs/String.h>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "../include/defense_vision_24/Box_Detector.hpp"

/*****************************************************************************
** Namespaces
*****************************************************************************/

namespace defense_vision_24
{
// 이진화 이미지 검출
cv::Mat Box_Detector::Binary(cv::Mat& img, int val[])
{
  // 복사 이미지 HSV로 변환
  cv::Mat hsvImg;
  cv::cvtColor(img, hsvImg, cv::COLOR_BGR2HSV);

  // Gaussian blur로 반사율 쥴이기
  cv::Mat blurredImage;
  cv::GaussianBlur(hsvImg, blurredImage, cv::Size(5, 5), 0);

  cv::Mat mask = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));
  cv::erode(blurredImage, blurredImage, mask, cv::Point(-1, -1), 1);

  // HSV 이미지를 사용하여 범위 내의 색상을 임계값으로 설정
  cv::Scalar lower(val[0], val[1], val[2]);
  cv::Scalar upper(val[3], val[4], val[5]);
  cv::Mat output_binary;
  cv::inRange(blurredImage, lower, upper, output_binary);

  return output_binary;
}

// Canny 엣지 검출
cv::Mat Box_Detector::canny_edge(cv::Mat& binary_img)
{
  // 가우시안 블러된 이미지에 Canny 엣지 검출 적용
  cv::Mat blurred_img;
  cv::GaussianBlur(binary_img, blurred_img, cv::Size(5, 5), 1.5);

  cv::Mat output_edge;
  double lower_thresh = 100;  // 하위 임계값 (낮을수록 더 많은 엣지를 찾음)
  double upper_thresh = 200;  // 상위 임계값 (높을수록 강한 엣지만 검출)
  int aperture_size = 3;      // 소벨 필터 커널 크기

  cv::Canny(blurred_img, output_edge, lower_thresh, upper_thresh, aperture_size);

  return output_edge;
}

// 선 관심 영역 설정 함수
cv::Mat Box_Detector::region_of_interest(cv::Mat& edge_img, int val[])
{
  // 이미지 크기와 같은 빈 이미지 생성
  cv::Mat output_roi;
  cv::Mat img_mask = cv::Mat::zeros(edge_img.size(), edge_img.type());  // height, width. CV_8UC1

  cv::Point points[4];                    // 관심 영역의 꼭지점 좌표 설정
  points[0] = cv::Point(val[0], val[1]);  // 좌측 하단 꼭지점
  points[1] = cv::Point(val[2], val[3]);  // 우측 하단 꼭지점
  points[2] = cv::Point(val[4], val[5]);  // 우측 상단 꼭지점
  points[3] = cv::Point(val[6], val[7]);  // 좌측 상단 꼭지점

  // 다각형이 채워질 대상 이미지, 꼭지점을 지정하는 포인터 배열, 꼭지점 수, 다각형의 수, 색상
  cv::fillConvexPoly(img_mask, points, 4, cv::Scalar(255, 0, 0));  // 결과이미지
  cv::bitwise_and(edge_img, img_mask, output_roi);  // 이미지와 마스크를 AND 연산하여 관심 영역만 추출

  return output_roi;
}

// hough 변환 직선 검출
std::vector<cv::Vec4i> Box_Detector::houghlines(cv::Mat roi_img)
{
  // 확률적용 허프변환 직선 검출
  std::vector<cv::Vec4i> output_houghline;
  cv::HoughLinesP(roi_img, output_houghline, 1, CV_PI / 180, 50, 30, 60);

  return output_houghline;
}

// 두 선분의 교차점을 확인하고 교차점을 반환하는 함수
bool Box_Detector::checkLineIntersection(cv::Vec4i l1, cv::Vec4i l2, cv::Point2f& intersection)
{
  // 첫 번째 선분의 두 점 (x1, y1) 및 (x2, y2) 좌표
  float x1 = l1[0], y1 = l1[1], x2 = l1[2], y2 = l1[3];

  // 두 번째 선분의 두 점 (x3, y3) 및 (x4, y4) 좌표
  float x3 = l2[0], y3 = l2[1], x4 = l2[2], y4 = l2[3];

  // 선분들의 방향 벡터의 행렬식 계산
  float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

  // 선분들이 평행하여 교차하지 않는 경우
  if (denom == 0)
  {
    return false;
  }

  // 교차점의 x, y 좌표 계산
  float intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
  float intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;

  // 교차점을 cv::Point2f 객체로 설정
  intersection = cv::Point2f(intersect_x, intersect_y);

  // 교차점이 두 선분의 범위 내에 있는지 확인
  if (intersect_x < std::min(x1, x2) || intersect_x > std::max(x1, x2) || intersect_x < std::min(x3, x4) ||
      intersect_x > std::max(x3, x4) || intersect_y < std::min(y1, y2) || intersect_y > std::max(y1, y2) ||
      intersect_y < std::min(y3, y4) || intersect_y > std::max(y3, y4))
  {
    return false;  // 범위 내에 없는 경우
  }

  return true;  // 교차점이 두 선분의 범위 내에 있는 경우
}

// 두 점 사이의 거리 계산
float Box_Detector::distance(const cv::Point2f& p1, const cv::Point2f& p2)
{
  return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

cv::Point2f Box_Detector::findCenter(const cv::Rect& rect)
{
  cv::Point2f center(rect.x + rect.width / 2, rect.y + rect.height / 2);

  return center;
}

}  // namespace defense_vision_24