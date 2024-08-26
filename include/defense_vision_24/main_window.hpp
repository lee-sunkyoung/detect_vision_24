/**
 * @file /include/defense_vision_24/main_window.hpp
 *
 * @brief Qt based gui for %(package)s.
 *
 * @date November 2010
 **/
#ifndef defense_vision_24_MAIN_WINDOW_H
#define defense_vision_24_MAIN_WINDOW_H

/*****************************************************************************
** Includes
*****************************************************************************/

#include <QMainWindow>
#include "ui_main_window.h"
#include "qnode.hpp"

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <vector>             // 벡터 관련 헤더 파일
#include "std_msgs/String.h"  // ROS 표준 메시지 타입을 사용하는 헤더 파일
#include <opencv2/opencv.hpp> // OpenCV 라이브러리의 기본 헤더 파일

#include "Box_Detector.hpp"

/*****************************************************************************
** Namespace
*****************************************************************************/

namespace defense_vision_24
{
  /*****************************************************************************
  ** Interface [MainWindow]
  *****************************************************************************/
  /**
   * @brief Qt central, all operations relating to the view part here.
   */
  class MainWindow : public QMainWindow
  {
    Q_OBJECT

  public:
    MainWindow(int argc, char **argv, QWidget *parent = 0);
    ~MainWindow();

    cv::Mat clone_mat; // 원본 이미지 복사
    cv::Mat frame;     // 프레임
    cv::Mat blob;      // 블롭
    cv::Mat red_img;
    cv::Mat dire_img;
    cv::Mat red_roi_img;

    int roi[8];
    int value_hsv[6];
    cv::Point2f center;
    cv::Point2f red_center;

    bool in_center_x;
    bool in_center_y;

    bool exact_x;
    bool exact_y;

    bool dire;
    bool dp;
    float slope = -1;

    bool isOverlapping; // 겹침 여부 플래그
    Box_Detector box_detector;

  public Q_SLOTS:
    void slotUpdateImg();

    void set_yolo();
    bool isRectOverlapping(const cv::Rect &rect1, const cv::Rect &rect2); // 사각형 겹침 여부 확인 함수

    void yolo_xy(const cv::Rect &rect);
    void exact_xy();
    void direction();
    void depth();

    std::vector<cv::Point2f> findClosestPoints(const std::vector<cv::Point2f> &points);
    void Find_Binary_img(cv::Mat &img);

  private:
    Ui::MainWindowDesign ui;
    QNode qnode;
  };

} // namespace defense_vision_24

#endif // defense_vision_24_MAIN_WINDOW_H
