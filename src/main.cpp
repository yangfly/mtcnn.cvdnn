#include "tool.hpp"

void demo(bool fdet1 = false, bool fp16 = false) {
  Mtcnn mtcnn("../models", fdet1, fp16);
  cv::Mat image = cv::imread("../images/test.jpg");
  std::vector<BBox> bboxes = mtcnn.Detect(image);
  cv::Mat canvas = imdraw(image, bboxes);
  cv::imshow("mtcnn face detector", canvas);
  cv::waitKey(0);
}

int main() {
  //shrink_models();
  // performance(true, false);
  demo(true, false);
  return 0;
}
