#include <iostream>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include "mtcnn.h"
#include "helper.hpp"

cv::Mat imdraw(const cv::Mat im, const std::vector<BBox> & bboxes)
{
  cv::Mat canvas = im.clone();
  for (const auto & bbox : bboxes) {
    cv::rectangle(canvas, cv::Rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1),
      cv::Scalar(255, 0, 0), 2);
    for (int i = 0; i < 5; i++) {
      cv::circle(canvas, cv::Point((int)bbox.fpoints[i], (int)bbox.fpoints[i + 5]),
        5, cv::Scalar(0, 255, 0), -1);
    }
    cv::putText(canvas, std::to_string(bbox.score), cv::Point(bbox.x1, bbox.y1),
      cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  }
  return canvas;
}

void performance(bool fdet1, bool fp16, int ntimes = 100) {
  Mtcnn mtcnn("../models", fdet1, fp16);
  cv::Mat image = cv::imread("../images/sample.jpg");
  std::string precise = fp16 ? "Float16" : "Float32";
  std::string model = fdet1 ? "Fast Pnet" : "Official Pnet";
  std::cout << "\nBenchmark with [" << model << "] on [" << precise
    << "] (" << ntimes << " times average)\n" << std::endl;

  std::cout << "  ------------------ With LNet -----------------\n" << std::endl;
  Timer timer;
  mtcnn.precise_landmark = true;
  for (int i = 0; i < ntimes; i++) {
    timer.tic();
    mtcnn.Detect(image);
    timer.toc();
  }
  std::cout << "  CPU Info: " << cpu_info() << std::endl;
  std::cout << "  Image Shape: (" << image.cols << ", " << image.rows
    << ", " << image.channels() << ")" << std::endl;
  std::cout << "  Performance : " << 1000 / timer.avg() << " fps" << std::endl;
  std::cout << "  Detect Time: " << timer.avg() << " ms\n" << std::endl;

  std::cout << "  ----------------- Without LNet ---------------\n" << std::endl;
  timer.reset();
  mtcnn.precise_landmark = false;
  for (int i = 0; i < ntimes; i++) {
    timer.tic();
    mtcnn.Detect(image);
    timer.toc();
  }
  std::cout << "  CPU Info: " << cpu_info() << std::endl;
  std::cout << "  Image Shape: (" << image.cols << ", " << image.rows
    << ", " << image.channels() << ")" << std::endl;
  std::cout << "  Performance : " << 1000 / timer.avg() << " fps" << std::endl;
  std::cout << "  Detect Time: " << timer.avg() << " ms\n" << std::endl;
}

#include <fstream>
void fddb_detect(std::string name = "mtcnn") {
  Mtcnn mtcnn("../models", true, false);
  mtcnn.thresholds[0] = 0.6f;
  mtcnn.thresholds[1] = 0.7f;
  mtcnn.thresholds[2] = 0.7f;
  mtcnn.face_min_size = 20;
  mtcnn.face_max_size = 2000;
  std::string fddb_root = "D:/vcodes/win-ncnn/FDDB";
  std::string img_root = fddb_root + "/images/";
  std::ifstream img_list(fddb_root + "/imList.txt");
  std::ofstream dets(fddb_root + "/dets/" + name + ".txt");
  Timer timer;
  std::string line;
  int cnt = 0;
  while (getline(img_list, line)) {
    cnt += 1;
    std::cout << "\r" << cnt << ": " << line << "\t" << std::flush;
    cv::Mat image = cv::imread(img_root + line + ".jpg");
    timer.tic();
    std::vector<BBox> bboxes = mtcnn.Detect(image);
    timer.toc();
    dets << line << std::endl;
    dets << bboxes.size() << std::endl;
    for (auto & bbox : bboxes) {
      dets << bbox.x1 << " " << bbox.y1 << " ";
      dets << bbox.x2 - bbox.x1 << " " << bbox.y2 - bbox.y1 << " ";
      dets << bbox.score << std::endl;
    }
  }
  std::cout << "\ndetect time: " << timer.avg() << " ms" << std::endl;
}

void shrink_models() {
  using namespace std::experimental::filesystem;
  v1::path fp32_root("../models/fp32");
  v1::path fp16_root("../models/fp16");
  if (v1::exists(fp16_root))
    v1::remove_all(fp16_root);
  v1::create_directory(fp16_root);
  CV_Assert(v1::exists(fp16_root));
  std::vector<cv::String> types = { "Convolution", "InnerProduct", "PReLU" };
  for (auto & p : v1::directory_iterator(fp32_root)) {
    std::cout << "convert model: " << p.path().filename() << std::endl;
    cv::dnn::shrinkCaffeModel(p.path().string(),
      (fp16_root.string() / p.path().filename()).string(), types);
  }
}
