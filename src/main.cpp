#include <iostream>
#include <opencv2/opencv.hpp>
#include "mtcnn.h"
#include "helper.h"

using namespace std;
using namespace cv;
using namespace face;

Mat imdraw(const Mat im, const vector<BBox> & bboxes)
{
  Mat canvas = im.clone();
  for (const auto & bbox : bboxes) {
    rectangle(canvas, Rect(bbox.x1, bbox.y1, bbox.x2-bbox.x1, bbox.y2-bbox.y1), Scalar(255, 0, 0), 2);
    for (int i = 0; i < 5; i++) {
      circle(canvas, Point((int)bbox.fpoints[i], (int)bbox.fpoints[i+5]), 5, Scalar(0, 255, 0), -1);
    }
    putText(canvas, to_string(bbox.score), Point(bbox.x1, bbox.y1),
      FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, LINE_AA);
  }
  return canvas;
}

void demo() {
  Mtcnn mtcnn("../models");
  Mat image = imread("../sample.jpg");
  vector<BBox> bboxes = mtcnn.Detect(image);
  Mat canvas = imdraw(image, bboxes);
  imshow("mtcnn face detector", canvas);
  cv::waitKey(0);
}

void performance(bool lnet = true, int ntimes = 100) {
  Mtcnn mtcnn("../models", lnet);
  Mat image = imread("../sample.jpg");
  Timer timer;
  for (int i = 0; i < ntimes; i++) {
    timer.tic();
    mtcnn.Detect(image);
    timer.toc();
  }
  string disc_head = lnet ? "With LNet" : "Without LNet";
  string disc_pad = "=============";
  cout << disc_pad << " " << disc_head << " " << disc_pad << endl;
  cout << "cpu info: " << cpu_info() << endl;
  cout << "image shape: (" << image.cols << ", " << image.rows << ", " << image.channels() << ")" << endl;
  cout << "performance : " << 1000 / timer.avg() << " fps" << endl;
  cout << "detect time: " << timer.avg() << " ms" << endl;
}

#include <fstream>
void fddb_detect(const string name = "mtcnn") {
  Mtcnn mtcnn("../models", false);
  mtcnn.thresholds[0] = 0.6f;
  mtcnn.thresholds[1] = 0.7f;
  mtcnn.thresholds[2] = 0.7f;
  mtcnn.face_min_size = 20;
  mtcnn.face_max_size = 2000;
  string fddb_root = "D:/vcodes/win-ncnn/FDDB";
  string img_root = fddb_root + "/images/";
  ifstream img_list(fddb_root + "/imList.txt");
  ofstream dets(fddb_root + "/dets/" + name + ".txt");
  Timer timer;
  string line;
  int cnt = 0;
  while (getline(img_list, line)) {
    cnt += 1;
    cout << "\r" << cnt << ": " << line << "\t" << flush;
    Mat image = imread(img_root + line + ".jpg");
    timer.tic();
    vector<BBox> bboxes = mtcnn.Detect(image);
    timer.toc();
    dets << line << endl;
    dets << bboxes.size() << endl;
    for (auto & bbox : bboxes) {
      dets << bbox.x1 << " " << bbox.y1 << " ";
      dets << bbox.x2 - bbox.x1 << " " << bbox.y2 - bbox.y1 << " ";
      dets << bbox.score << endl;
    }
  }
  cout << "\ndetect time: " << timer.avg() << " ms" << endl;
}

int main() {
  performance(true);
  performance(false);
  //fddb_detect("cvdnn");
  demo();
  return 0;
}
