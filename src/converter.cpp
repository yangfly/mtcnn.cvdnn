#include <iostream>
#include <opencv2/dnn.hpp>
using namespace cv;

int main() {
  String name = "../models/det";
  std::vector<String> types = { "Convolution", "InnerProduct", "PReLU" };
  for (char c = '1'; c <= '4'; c++) {
    dnn::shrinkCaffeModel(
      name + c + ".caffemodel",
      name + c + ".fp16.caffemodel",
      types);
  }
  return 0;
}