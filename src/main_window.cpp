/**
 * @file /src/main_window.cpp
 *
 * @brief Implementation for the qt gui.
 *
 * @date February 2011
 **/
/*****************************************************************************
** Includes
*****************************************************************************/

#include <QtGui>
#include <QMessageBox>
#include <iostream>
#include "../include/defense_vision_24/main_window.hpp"
#include <std_msgs/Float32MultiArray.h>
#include <cmath>
#define PI 3.1415
/*****************************************************************************
** Namespaces
*****************************************************************************/

constexpr float CONFIDENCE_THRESHOLD = 0.45; // 확률 경계값
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 1;

const cv::Scalar colors[] = {{0, 255, 255}, {255, 255, 0}, {0, 255, 0}, {255, 0, 0}};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

const int cam_x = 640;
const int cam_y = 360;
const float cam_center_x = 320.0;
const float cam_center_y = 180.0;

namespace defense_vision_24
{
  using namespace Qt;

  /*****************************************************************************
  ** Implementation [MainWindow]
  *****************************************************************************/

  MainWindow::MainWindow(int argc, char **argv, QWidget *parent) : QMainWindow(parent), qnode(argc, argv)
  {
    ui.setupUi(this); // Calling this incidentally connects all ui's triggers to on_...() callbacks in this class.

    setWindowIcon(QIcon(":/images/icon.png"));

    qnode.init();

    roi[0] = 0;
    roi[1] = cam_y;
    roi[2] = 0;
    roi[3] = 0;
    roi[4] = cam_x;
    roi[5] = 0;
    roi[6] = cam_x;
    roi[7] = cam_y;

    qnode.mvis_pub.mode = "";
    qnode.mvis_pub.grip = false;
    qnode.mvis_pub.difference[0] = 0;
    qnode.mvis_pub.difference[1] = 0;
    qnode.mvis_pub.difference[2] = 0;
    qnode.mvis_pub.angle[0] = 0;
    qnode.mvis_pub.angle[1] = 0;
    qnode.mvis_pub.angle[2] = 0;
    qnode.mvis_pub.turning_direction = 0;

    in_center_x = false;
    in_center_y = false;

    exact_x = false;
    exact_y = false;

    dire = false;
    dp = false;
    qnode.depth = 0.0;

    QObject::connect(&qnode, SIGNAL(rosShutdown()), this, SLOT(close()));
    QObject::connect(&qnode, SIGNAL(sigRcvImg()), this, SLOT(slotUpdateImg()));
  }

  MainWindow::~MainWindow()
  {
  }

  /*****************************************************************************
  ** Functions
  *****************************************************************************/
  void MainWindow::slotUpdateImg()
  {
    if (qnode.mvis_sub.mode == "run")
    {
      if (!qnode.imgRaw)
      {
        std::cerr << "Error : qnode not exist" << std::endl;
        return;
      }
      clone_mat = qnode.imgRaw->clone();                                           // 원본 이미지 복사
      cv::resize(clone_mat, clone_mat, cv::Size(640, 360), 0, 0, cv::INTER_CUBIC); // 이미지 크기 조정

      set_yolo();

      Find_Binary_img(clone_mat);

      QImage colorQImage(dire_img.data, dire_img.cols, dire_img.rows, dire_img.step, QImage::Format_RGB888);
      ui.red->setPixmap(QPixmap::fromImage(colorQImage));

      QImage redQImage(red_img.data, red_img.cols, red_img.rows, red_img.step, QImage::Format_RGB888);
      ui.test->setPixmap(QPixmap::fromImage(redQImage));

      QImage blackQImage(red_roi_img.data, red_roi_img.cols, red_roi_img.rows, red_roi_img.step,
                         QImage::Format_Grayscale8);
      ui.black->setPixmap(QPixmap::fromImage(blackQImage));

      qnode.mani_vision_pub.publish(qnode.mvis_pub);

      delete qnode.imgRaw; // 동적 할당된 원본 이미지 메모리 해제
      qnode.imgRaw = NULL;
      qnode.isreceived = false; // 이미지 수신 플래그 재설정
    }
  }

  void MainWindow::set_yolo()
  {
    frame = clone_mat.clone(); // 프레임 복제
    if (frame.empty())
    {
      std::cerr << "Error: cv::Mat is empty" << std::endl;
      return;
    }
    auto output_names = qnode.net.getUnconnectedOutLayersNames(); // 출력 레이어 이름 가져오기
    std::vector<cv::Mat> detections;                              // 감지 결과를 담을 벡터

    cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false,
                           CV_32F); // 이미지를 blob으로 변환
    qnode.net.setInput(blob);       // 네트워크 입력 설정

    auto dnn_start = std::chrono::steady_clock::now(); // 딥러닝 전파 시작 시간 측정
    qnode.net.forward(detections, output_names);       // 딥러닝 전파 실행
    auto dnn_end = std::chrono::steady_clock::now();   // 딥러닝 전파 종료 시간 측정

    // 감지된 객체의 박스와 점수를 저장할 벡터들
    std::vector<int> indices[NUM_CLASSES];
    std::vector<cv::Rect> boxes[NUM_CLASSES];
    std::vector<float> scores[NUM_CLASSES];

    // 각 detection에 대해 반복
    for (auto &output : detections)
    {
      const auto num_boxes = output.rows;
      for (int i = 0; i < num_boxes; i++)
      {
        auto x = output.at<float>(i, 0) * frame.cols;                // 중심 x 좌표 계산
        auto y = output.at<float>(i, 1) * frame.rows;                // 중심 y 좌표 계산
        auto width = output.at<float>(i, 2) * frame.cols;            // 너비 계산
        auto height = output.at<float>(i, 3) * frame.rows;           // 높이 계산
        cv::Rect rect(x - width / 2, y - height / 2, width, height); // 박스 생성

        // 클래스에 대한 반복
        for (int c = 0; c < NUM_CLASSES; c++)
        {
          auto confidence = *output.ptr<float>(i, 5 + c); // 확률 가져오기
          if (confidence >= CONFIDENCE_THRESHOLD)         // 신뢰도 임계값 이상인 경우
          {
            boxes[c].push_back(rect);        // 박스 추가
            scores[c].push_back(confidence); // 점수 추가
          }
        }
      }
    }

    // 각 클래스에 대해 비최대 억제 수행
    for (int c = 0; c < NUM_CLASSES; c++)
      cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

    // 비최대 억제 후의 각 박스에 대해 처리
    for (int c = 0; c < NUM_CLASSES; c++)
    {
      for (size_t i = 0; i < indices[c].size(); ++i)
      {
        const auto color = colors[c % NUM_COLORS]; // 클래스별 색상 선택
        auto idx = indices[c][i];
        const auto &rect = boxes[c][idx]; // 현재 박스

        // 같은 클래스의 중복되는 박스 확인
        isOverlapping = false;
        if (indices[c].size() != 0)
        {
          for (size_t j = 0; j < indices[c].size(); ++j)
          {
            if (j != i)
            {
              auto idx2 = indices[c][j];
              const auto &rect2 = boxes[c][idx2];
              if (isRectOverlapping(rect, rect2))
              {
                // 중복되는 박스가 있으면 크기를 비교
                if (rect2.area() < rect.area())
                {
                  // 다른 박스가 더 작으면 중복 표시 후 종료
                  isOverlapping = true;
                  break;
                }
                else
                {
                  // 현재 박스가 더 작으면 건너뜀
                  continue;
                }
              }
            }
          }
        }

        // 작은 박스와 중복되지 않는 경우에만 박스 그리기
        if (!isOverlapping)
        {
          cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color,
                        3); // 박스그리기

          // 라벨과 점수 출력
          std::ostringstream label_ss;
          label_ss << qnode.class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
          auto label = label_ss.str();

          int baseline;
          auto label_bg_sz =
              cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline); // 라벨크기 계산
          cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10),
                        cv::Point(rect.x + label_bg_sz.width, rect.y), color,
                        cv::FILLED); // 라벨 배경
          cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                      cv::Scalar(0, 0, 0)); // 라벨 텍스트

          ///////////////////////////////////////////////////////////
          roi[0] = rect.x + 30;
          roi[1] = rect.y + rect.height - 30;
          roi[2] = rect.x + 30;
          roi[3] = rect.y + 30;
          roi[4] = rect.x + rect.width - 30;
          roi[5] = rect.y + 30;
          roi[6] = rect.x + rect.width - 30;
          roi[7] = rect.y + rect.height - 30;

          center = box_detector.findCenter(rect);
          if (in_center_x == false || in_center_y == false)
            yolo_xy(rect);

          else if ((in_center_x == true && in_center_y == true) && (exact_x == false || exact_y == false))
            exact_xy();

          else if ((exact_x == true && exact_y == true) && (dire == false))
            direction();

          else if (dire == true && dp == false)
            depth();
          else if (boxes[0].empty() && dp == false)
          {
            // 감지된 객체가 없는 경우의 처리
            std::cout << "No objects detected." << std::endl;
            qnode.mvis_pub.mode = "run";
            qnode.mvis_pub.grip = false;
            qnode.mvis_pub.difference[0] = 0;
            qnode.mvis_pub.difference[1] = 0;
            qnode.mvis_pub.difference[2] = 0;
            qnode.mvis_pub.angle[0] = 0;
            qnode.mvis_pub.angle[1] = 0;
            qnode.mvis_pub.angle[2] = 0;
            qnode.mvis_pub.turning_direction = 0;
          }
          else if (dp == true)
          {
            qnode.mvis_pub.mode = "finish";
            qnode.mvis_pub.grip = false;
            qnode.mvis_pub.difference[0] = 0;
            qnode.mvis_pub.difference[1] = 0;
            qnode.mvis_pub.difference[2] = 0;
            qnode.mvis_pub.angle[0] = 0;
            qnode.mvis_pub.angle[1] = 0;
            qnode.mvis_pub.angle[2] = 0;
            qnode.mvis_pub.turning_direction = 0;
          }
        }
      }
    }
    // yolo 이미지
    QImage yolo_im((const unsigned char *)(frame.data), frame.cols, frame.rows, QImage::Format_RGB888);
    ui.label->setPixmap(QPixmap::fromImage(yolo_im));
  }

  bool MainWindow::isRectOverlapping(const cv::Rect &rect1, const cv::Rect &rect2)
  {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    if (x1 < x2 && y1 < y2)
      return true;
    else
      return false;
  }
  //////////////////////////////////////////////////////////////////////////////////
  void MainWindow::yolo_xy(const cv::Rect &rect)
  {
    float object_center_x = rect.x + rect.width / 2;
    float object_center_y = rect.y + rect.height / 2;

    // 음수면 오른쪽, 아래쪽
    float diff_x = object_center_x - cam_center_x;
    float diff_y = object_center_y - cam_center_y;
    std::cout << "yolo_xy : " << diff_x << " " << diff_y << std::endl;

    if (diff_x < 20 && diff_x > -20)
    {
      in_center_x = true;
    }
    else
    {
      in_center_x = false;
    }

    if (diff_y - 90 < 20 && diff_y - 90 > -20)
    {
      in_center_y = true;
    }
    else
    {
      in_center_y = false;
    }
    qnode.mvis_pub.mode = "run";
    qnode.mvis_pub.grip = false;
    qnode.mvis_pub.difference[0] = diff_x;
    qnode.mvis_pub.difference[1] = 0;
    qnode.mvis_pub.difference[2] = diff_y - 90;
    qnode.mvis_pub.angle[0] = 0;
    qnode.mvis_pub.angle[1] = 0;
    qnode.mvis_pub.angle[2] = 0;
    qnode.mvis_pub.turning_direction = 0;
  }

  ////////////////////////////////////////////////////////////////////////////////////////

  void MainWindow::exact_xy()
  {
    red_img = clone_mat.clone();
    if (red_img.empty())
    {
      std::cerr << "Error: cv::Mat is empty" << std::endl;
      return;
    }
    int red_HSV_value[6] = {90, 46, 90, 255, 255, 255};
    cv::Mat red_plane = box_detector.Binary(red_img, red_HSV_value); // 빨간색 영역을 이진화
    red_roi_img = box_detector.region_of_interest(red_plane, roi);   // ROI 설정
    std::vector<std::vector<cv::Point>> contours;                    // 컨투어 찾기
    cv::findContours(red_roi_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
      std::cerr << "Error: No contours found" << std::endl;
      return; // 빨간색 영역이 없으면 함수 종료
    }

    // 가장 큰 컨투어를 찾기 위한 변수
    double max_area = 0;
    std::vector<cv::Point> largest_contour;

    for (const auto &contour : contours)
    {
      double area = cv::contourArea(contour);
      if (area > max_area)
      {
        max_area = area;
        largest_contour = contour;
      }
    }
    if (!contours.empty())
    {
      cv::Rect bounding_box = cv::boundingRect(largest_contour);              // 가장 큰 컨투어에 대한 바운딩 박스 계산
      red_center = box_detector.findCenter(bounding_box);                     // 바운딩 박스의 중심점 계산
      cv::circle(red_img, red_center, 10, cv::Scalar(0, 0, 255), cv::FILLED); // 중심점에 빨간색 원 표시

      /////////////////////////////////////////////////////////////////////
      float object_center_x = red_center.x;
      float object_center_y = red_center.y;

      // 음수면 오른쪽, 아래쪽
      float diff_x = object_center_x - cam_center_x;
      float diff_y = object_center_y - cam_center_y;
      std::cout << "exact_xy : " << diff_x << " " << diff_y << std::endl;

      if (diff_x < 15 && diff_x > -15)
      {
        exact_x = true;
      }
      else
      {
        exact_x = false;
      }

      if (diff_y - 90 < 15 && diff_y - 90 > -15)
      {
        exact_y = true;
      }
      else
      {
        exact_y = false;
      }

      qnode.mvis_pub.mode = "run";
      qnode.mvis_pub.grip = false;
      qnode.mvis_pub.difference[0] = diff_x;
      qnode.mvis_pub.difference[1] = 0;
      qnode.mvis_pub.difference[2] = diff_y - 90;
      qnode.mvis_pub.angle[0] = 0;
      qnode.mvis_pub.angle[1] = 0;
      qnode.mvis_pub.angle[2] = 0;
      qnode.mvis_pub.turning_direction = 0;
    }
  }

  void MainWindow::direction()
  {
    dire_img = clone_mat.clone();
    if (dire_img.empty())
    {
      std::cerr << "Error: cv::Mat is empty" << std::endl;
      return;
    }
    cv::Mat can_img = box_detector.canny_edge(dire_img);                  // 에지 검출
    cv::Mat roi_img = box_detector.region_of_interest(can_img, roi);      // ROI 설정
    std::vector<cv::Vec4i> houghlines = box_detector.houghlines(roi_img); // 허프 변환으로 선 검출

    std::vector<cv::Point2f> intersections;

    // 선분들 사이의 교차점을 찾고 교차점에 원 표시
    for (size_t i = 0; i < houghlines.size(); i++)
    {

      for (size_t j = i + 1; j < houghlines.size(); j++)
      {
        cv::Vec4i l = houghlines[i];
        cv::line(dire_img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        cv::Vec4i l1 = houghlines[i];
        cv::Vec4i l2 = houghlines[j];

        cv::Point2f intersection;
        if (box_detector.checkLineIntersection(l1, l2, intersection))
        {
          if (intersections.size() <= 16) // 교차점이 16개 이하일 때만 추가
          {
            intersections.push_back(intersection);                                     // 교차점을 저장
            cv::circle(dire_img, intersection, 10, cv::Scalar(0, 0, 255), cv::FILLED); // 교차점에 파란색 원 표시
          }
          else
          {
            break; // 16개 교차점을 찾으면 더 이상 탐색하지 않음
          }
        }
      }
    }
    if (houghlines.size() == 0)
    {
      return;
    }

    // // 교차점 중 가장 중심에 가까운 4개 점 찾기
    // std::vector<cv::Point2f> closestPoints = findClosestPoints(intersections);

    // // 중심에 가까운 4개 점을 빨간색 원으로 표시
    // for (const auto &point : closestPoints)
    // {
    //   cv::circle(dire_img, point, 5, cv::Scalar(255, 0, 0), cv::FILLED); // 빨간색 원으로 표시
    // }

    cv::Point2f point1 = {0, 0};

    for (const auto &point : intersections)
    {
      if (point.x < 320 && point.y < 180 + 90 - 5)
      {
        point1 = point;
        // 기울기
        if (point1.x < 320 && point1.y < 180 + 90 && point1.x != 0 && point1.y != 0)
          slope = (180 + 90 - point1.y) / (320 - point1.x);

        if (320 == point1.x)
        {
          std::cerr << "Error: Vertical line detected, slope is undefined" << std::endl;
          return;
        }
        if (slope == 0)
        {
          return;
        }

        float angle;
        int turn_direction; // 1 for left, 2 for right side.
        red_center.x = 320;
        red_center.y = 180 + 90;
        if (slope > 0 && slope != 0.5625)
        {
          angle = 45 - std::atan(slope) * 180.0 / PI;

          std::cout << "angle : " << angle << std::endl;
          cv::line(dire_img, point1, red_center, cv::Scalar(255, 255, 255), 2); // 빨간색 선 표시

          if (angle > 0)
          {
            turn_direction = 2; // 오른쪽으로 회전
            std::cout << "right" << std::endl;
          }

          else
          {
            turn_direction = 1; // 왼쪽으로 회전
            std::cout << "left" << std::endl;
          }

          if (angle < 2 && angle > -2)
          {
            dire = true;
          }
          else

          {
            dire = false;
          }
          qnode.mvis_pub.angle[0] = angle;
          qnode.mvis_pub.turning_direction = turn_direction;
        }
      }
    }
    ///////////////////////////////////////////////////////////////
    qnode.mvis_pub.mode = "run";
    qnode.mvis_pub.grip = false;
    qnode.mvis_pub.difference[0] = 0;
    qnode.mvis_pub.difference[1] = 0;
    qnode.mvis_pub.difference[2] = 0;

    qnode.mvis_pub.angle[1] = 0;
    qnode.mvis_pub.angle[2] = 0;
  }

  void MainWindow::depth()
  {

    if (qnode.depth <= 200)
    {
      qnode.mvis_pub.mode = "finish";
      qnode.mvis_pub.grip = false;
      qnode.mvis_pub.difference[0] = 0;
      qnode.mvis_pub.difference[1] = 0;
      qnode.mvis_pub.difference[2] = 0;
      qnode.mvis_pub.angle[0] = 0;
      qnode.mvis_pub.angle[1] = 0;
      qnode.mvis_pub.angle[2] = 0;
      qnode.mvis_pub.turning_direction = 0;
      std::cout << "end" << std::endl;
      dp = true;
    }
    else if (qnode.depth > 200)
    {
      qnode.mvis_pub.mode = "run";
      qnode.mvis_pub.grip = false;
      qnode.mvis_pub.difference[0] = 0;
      qnode.mvis_pub.difference[1] = qnode.depth;
      qnode.mvis_pub.difference[2] = 0;
      qnode.mvis_pub.angle[0] = 0;
      qnode.mvis_pub.angle[1] = 0;
      qnode.mvis_pub.angle[2] = 0;
      qnode.mvis_pub.turning_direction = 0;
      std::cout << "depth : " << qnode.depth << std::endl;
    }
  }

  // 가장 중심에 가까운 4개의 점을 찾기 위한 함수
  std::vector<cv::Point2f> MainWindow::findClosestPoints(const std::vector<cv::Point2f> &points)
  {
    std::vector<std::pair<float, cv::Point2f>> distances;

    // Calculate distance of each point from the center
    for (const auto &point : points)
    {
      float distance = std::sqrt(std::pow(point.x - 320, 2) + std::pow(point.y - 180, 2));
      distances.push_back(std::make_pair(distance, point));
    }

    // Sort based on distance
    std::sort(
        distances.begin(), distances.end(),
        [](const std::pair<float, cv::Point2f> &a, const std::pair<float, cv::Point2f> &b)
        { return a.first < b.first; });

    // Select the four closest points
    std::vector<cv::Point2f> closestPoints;
    for (int i = 0; i < std::min(4, static_cast<int>(distances.size())); ++i)
    {
      closestPoints.push_back(distances[i].second);
    }

    return closestPoints;
  }

  // 이진화 찾는 함수
  void MainWindow::Find_Binary_img(cv::Mat &img)
  {
    // 이진화
    cv::Mat F_Image = box_detector.Binary(img, value_hsv);

    // 슬라이더
    value_hsv[0] = ui.horizontalSlider_7->value();  // low h
    value_hsv[1] = ui.horizontalSlider_8->value();  // low s
    value_hsv[2] = ui.horizontalSlider_9->value();  // low v
    value_hsv[3] = ui.horizontalSlider_10->value(); // high h
    value_hsv[4] = ui.horizontalSlider_11->value(); // high s
    value_hsv[5] = ui.horizontalSlider_12->value(); // high v

    // 슬라이더 값
    ui.s_1->display(value_hsv[0]);
    ui.s_2->display(value_hsv[1]);
    ui.s_3->display(value_hsv[2]);
    ui.s_4->display(value_hsv[3]);
    ui.s_5->display(value_hsv[4]);
    ui.s_6->display(value_hsv[5]);

    // 이진 이미지를 표시
    //   QImage binaryQImage(F_Image.data, F_Image.cols, F_Image.rows, F_Image.step, QImage::Format_Grayscale8);
    //   ui.test2->setPixmap(QPixmap::fromImage(binaryQImage));
  }

} // namespace defense_vision_24
