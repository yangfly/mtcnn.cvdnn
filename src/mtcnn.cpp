#include <algorithm>  // std::min, std::max, std::sort
#include <opencv2/imgproc.hpp>
#include "mtcnn.h"
using namespace std;
using namespace face;

#include <limits>
int fix(float f)
{
  f *= 1 + numeric_limits<float>::epsilon();
  return static_cast<int>(f);
}

Mtcnn::Mtcnn(const string & model_dir, bool Lnet) :
  lnet(Lnet)
{
  // load models
  Pnet = cv::dnn::readNet(model_dir + "/det1.prototxt", model_dir + "/det1.fp16.caffemodel");
  Rnet = cv::dnn::readNet(model_dir + "/det2.prototxt", model_dir + "/det2.fp16.caffemodel");
  Onet = cv::dnn::readNet(model_dir + "/det3.prototxt", model_dir + "/det3.fp16.caffemodel");
  if (lnet)
    this->Lnet = cv::dnn::readNet(model_dir + "/det4.prototxt", model_dir + "/det4.fp16.caffemodel");
}

Mtcnn::~Mtcnn() {}

#include <iostream>
vector<BBox> Mtcnn::Detect(const cv::Mat & image)
{
  vector<_BBox> _bboxes = ProposalNetwork(image);
  RefineNetwork(image, _bboxes);
  OutputNetwork(image, _bboxes);
  if (precise_landmark && lnet)
    LandmarkNetwork(image, _bboxes);
  vector<BBox> bboxes;
  for (const _BBox & _bbox : _bboxes)
    bboxes.emplace_back(_bbox.base());
  return bboxes;
}

BBox Mtcnn::Landmark(const cv::Mat & image, BBox bbox) {
  vector<_BBox> _bboxes = { _BBox(bbox) };
  OutputNetwork(image, _bboxes);
  if (precise_landmark && lnet)
    LandmarkNetwork(image, _bboxes);
  if (!_bboxes.empty()) {
    return _bboxes[0].base();
  }
  else {
    return BBox();
  }
}

vector<float> Mtcnn::ScalePyramid(const int min_len)
{
  vector<float> scales;
  float max_scale = 12.0f / face_min_size;
  float min_scale = 12.0f / std::min<int>(min_len, face_max_size);
  for (float scale = max_scale; scale >= min_scale; scale *= scale_factor)
    scales.push_back(scale);
  return scales;
}

vector<Mtcnn::_BBox> Mtcnn::GetCandidates(const float scale, 
  const cv::Mat & conf_blob, const cv::Mat & loc_blob)
{
  int stride = 2;
  int cell_size = 12;
  float inv_scale = 1.0f / scale;
  vector<_BBox> condidates;

  const float* conf_data = conf_blob.ptr<float>(0, 1);
  vector<const float*> loc_data = {
    loc_blob.ptr<float>(0, 0),
    loc_blob.ptr<float>(0, 1),
    loc_blob.ptr<float>(0, 2),
    loc_blob.ptr<float>(0, 3)
  };

  int id = 0;
  for (int i = 0; i < conf_blob.size[2]; ++i)
    for (int j = 0; j < conf_blob.size[3]; ++j) {
      float score = conf_data[id];
      if (score >= thresholds[0]) {
        condidates.emplace_back();
        _BBox & _bbox = *condidates.rbegin();
        _bbox.score = score;
        _bbox.x1 = static_cast<int>(round((j * stride + 1) * inv_scale) - 1);
        _bbox.y1 = static_cast<int>(round((i * stride + 1) * inv_scale) - 1);
        _bbox.x2 = static_cast<int>(round((j * stride + cell_size) * inv_scale));
        _bbox.y2 = static_cast<int>(round((i * stride + cell_size) * inv_scale));
        for (int i = 0; i < 4; i++)
          _bbox.regs[i] = loc_data[i][id];
      }
      id++;
    }
  return condidates;
}

void Mtcnn::NonMaximumSuppression(std::vector<Mtcnn::_BBox> & _bboxes,
  const float threshold, const NMS_TYPE type) {
  if (_bboxes.size() <= 1)
    return;

  sort(_bboxes.begin(), _bboxes.end(),
    // Lambda function: descending order by score.
    [](const _BBox & x, const _BBox & y) -> bool { return x.score > y.score; });

  int keep = 0;  // index of candidates to be keeped
  while (keep < _bboxes.size()) {
    // pass maximun candidates.
    const _BBox & _max = _bboxes[keep++];
    int max_area = _max.area();
    // filter out overlapped candidates in the rest.
    for (int i = keep; i < _bboxes.size(); ) {
      // computer intersection.
      const _BBox & _bbox = _bboxes[i];
      int x1 = std::max<int>(_max.x1, _bbox.x1);
      int y1 = std::max<int>(_max.y1, _bbox.y1);
      int x2 = std::min<int>(_max.x2, _bbox.x2);
      int y2 = std::min<int>(_max.y2, _bbox.y2);
      float overlap = 0.f;
      if (x1 < x2 && y1 < y2) {
        int inter = (x2 - x1) * (y2 - y1);
        int outer;
        if (type == IoM)  // Intersection over Minimum
          outer = std::min<int>(max_area, _bbox.area());
        else  // Intersection over Union
          outer = max_area + _bbox.area() - inter;
        overlap = static_cast<float>(inter) / outer;
      }
      if (overlap > threshold)
        // erase overlapped candidate.
        _bboxes.erase(_bboxes.begin() + i);
      else
        i++;  // check next candidate.
    }
  }
}

void Mtcnn::BoxRegression(std::vector<Mtcnn::_BBox> & _bboxes, bool square)
{
  for (auto & _bbox : _bboxes) {
    // bbox regression.
    float w = static_cast<float>(_bbox.x2 - _bbox.x1);
    float h = static_cast<float>(_bbox.y2 - _bbox.y1);
    float x1 = _bbox.x1 + _bbox.regs[0] * w;
    float y1 = _bbox.y1 + _bbox.regs[1] * h;
    float x2 = _bbox.x2 + _bbox.regs[2] * w;
    float y2 = _bbox.y2 + _bbox.regs[3] * h;
    // expand bbox to square.
    if (square) {
      w = x2 - x1;
      h = y2 - y1;
      float maxl = std::max<float>(w, h);
      _bbox.x1 = static_cast<int>(round(x1 + (w - maxl) * 0.5f));
      _bbox.y1 = static_cast<int>(round(y1 + (h - maxl) * 0.5f));
      _bbox.x2 = _bbox.x1 + fix(maxl);
      _bbox.y2 = _bbox.y1 + fix(maxl);
    }
    else {
      _bbox.x1 = static_cast<int>(round(x1));
      _bbox.y1 = static_cast<int>(round(y1));
      _bbox.x2 = static_cast<int>(round(x2));
      _bbox.y2 = static_cast<int>(round(y2));
    }
  }
}

cv::Mat Mtcnn::PadCrop(const cv::Mat & image, const cv::Rect & roi)
{
  cv::Rect img_rect(0, 0, image.cols, image.rows);
  cv::Mat crop(roi.size(), CV_32FC3, cv::Scalar(0.f));
  cv::Rect inter_on_sample = roi & img_rect;
  if (! inter_on_sample.empty())
  {
    // shifting inter from image CS (coordinate system) to crop CS.
    cv::Rect inter_on_crop = inter_on_sample - roi.tl();
    image(inter_on_sample).copyTo(crop(inter_on_crop));
  }
  return crop;
}

vector<Mtcnn::_BBox> Mtcnn::ProposalNetwork(const cv::Mat & image)
{
  int min_len = std::min<int>(image.rows, image.cols);
  vector<float> scales = ScalePyramid(min_len);
  vector<cv::String> out_names = { "prob1", "conv4-2" };
  vector<_BBox> total_bboxes;
  for (float scale : scales) {
    int width = static_cast<int>(ceil(image.cols * scale));
    int height = static_cast<int>(ceil(image.rows * scale));
    cv::Mat input;
    cv::resize(image, input, cv::Size(width, height));
    vector<cv::Mat> out_blobs;
    Pnet.setInput(cv::dnn::blobFromImage(input, 1.f, cv::Size(), cv::Scalar(), false), "data");
    Pnet.forward(out_blobs, out_names);
    vector<_BBox> scale_bboxes = GetCandidates(scale, out_blobs[0], out_blobs[1]);
    // intra scale nms
    NonMaximumSuppression(scale_bboxes, 0.5f, IoU);
    if (!scale_bboxes.empty()) {
      total_bboxes.insert(total_bboxes.end(), scale_bboxes.begin(), scale_bboxes.end());
    }
  }
  // inter scale nms
  NonMaximumSuppression(total_bboxes, 0.7f, IoU);
  BoxRegression(total_bboxes, true);
  return total_bboxes;
}

void Mtcnn::RefineNetwork(const cv::Mat & image, vector<Mtcnn::_BBox> & _bboxes)
{
  if (_bboxes.empty())
    return;

  vector<cv::Mat> inputs;
  for (const _BBox & _bbox : _bboxes) {
    cv::Rect roi(_bbox.x1, _bbox.y1, _bbox.x2 - _bbox.x1, _bbox.y2 - _bbox.y1);
    cv::Mat crop = PadCrop(image, roi);
    cv::resize(crop, crop, cv::Size(24, 24));
    inputs.push_back(crop);
  }
  vector<cv::String> out_names = { "prob1", "fc5-2" };
  vector<cv::Mat> out_blobs;
  Rnet.setInput(cv::dnn::blobFromImages(inputs, 1.f, cv::Size(), cv::Scalar(), false), "data");
  Rnet.forward(out_blobs, out_names);
  const float* conf_data = out_blobs[0].ptr<float>(0);
  const float* loc_data = out_blobs[1].ptr<float>(0);
  vector<int> keep;
  for (int i = 0; i < _bboxes.size(); i++) {
    float score = conf_data[2 * i + 1];
    if (score >= thresholds[1]) {
      _bboxes[i].score = score;
      for (int j = 0; j < 4; j++)
        _bboxes[i].regs[j] = loc_data[4 * i + j];
      keep.push_back(i);
    }
  }
  if (keep.size() < _bboxes.size()) {
    for (int i = 0; i < keep.size(); i++) {
      if (i < keep[i])
        _bboxes[i] = _bboxes[keep[i]];
    }
    _bboxes.erase(_bboxes.begin() + keep.size(), _bboxes.end());
  }

  NonMaximumSuppression(_bboxes, 0.7f, IoU);
  BoxRegression(_bboxes, true);
}

void Mtcnn::OutputNetwork(const cv::Mat & image, vector<Mtcnn::_BBox> & _bboxes)
{
  if (_bboxes.empty())
    return;

  vector<cv::Mat> inputs;
  for (const _BBox & _bbox : _bboxes) {
    cv::Rect roi(_bbox.x1, _bbox.y1, _bbox.x2 - _bbox.x1, _bbox.y2 - _bbox.y1);
    cv::Mat crop = PadCrop(image, roi);
    cv::resize(crop, crop, cv::Size(48, 48));
    inputs.push_back(crop);
  }
  vector<cv::String> out_names = { "prob1", "fc6-2", "fc6-3" };
  vector<cv::Mat> out_blobs;
  Onet.setInput(cv::dnn::blobFromImages(inputs, 1.f, cv::Size(), cv::Scalar(), false), "data");
  Onet.forward(out_blobs, out_names);
  const float* conf_data = out_blobs[0].ptr<float>(0);
  const float* loc_data = out_blobs[1].ptr<float>(0);
  const float* kpt_data = out_blobs[2].ptr<float>(0);
  vector<int> keep;
  for (int i = 0; i < _bboxes.size(); i++) {
    float score = conf_data[2 * i + 1];
    if (score >= thresholds[2]) {
      _bboxes[i].score = score;
      for (int j = 0; j < 4; j++)
        _bboxes[i].regs[j] = loc_data[4 * i + j];
      // facial landmarks
      int w = _bboxes[i].x2 - _bboxes[i].x1;
      int h = _bboxes[i].y2 - _bboxes[i].y1;
      for (int j = 0; j < 5; j++) {
        _bboxes[i].fpoints[j] = kpt_data[10*i+j] * w + _bboxes[i].x1;
        _bboxes[i].fpoints[j + 5] = kpt_data[10*i+j+5] * h + _bboxes[i].y1;
      }
      keep.push_back(i);
    }
  }
  if (keep.size() < _bboxes.size()) {
    for (int i = 0; i < keep.size(); i++) {
      if (i < keep[i])
        _bboxes[i] = _bboxes[keep[i]];
    }
    _bboxes.erase(_bboxes.begin() + keep.size(), _bboxes.end());
  }

  BoxRegression(_bboxes, false);
  NonMaximumSuppression(_bboxes, 0.7f, IoM);
}

void Mtcnn::LandmarkNetwork(const cv::Mat & image, vector<Mtcnn::_BBox> & _bboxes)
{
  if (_bboxes.empty())
    return;

  vector<vector<cv::Mat>> inputs(5);
  vector<int> patchws;
  for (_BBox & _bbox : _bboxes) {
    int patchw = std::max<int>(_bbox.x2 - _bbox.x1, _bbox.y2 - _bbox.y1);
    patchw = fix(patchw * 0.25f);
    if (patchw % 2 == 1)
      patchw += 1;
    patchws.push_back(patchw);
    for (int i = 0; i < 5; i++) {
      _bbox.fpoints[i] = round(_bbox.fpoints[i]);
      _bbox.fpoints[i + 5] = round(_bbox.fpoints[i + 5]);
      int x1 = fix(_bbox.fpoints[i]) - patchw / 2;
      int y1 = fix(_bbox.fpoints[i + 5]) - patchw / 2;
      cv::Rect roi(x1, y1, patchw, patchw);
      cv::Mat crop = PadCrop(image, roi);
      cv::resize(crop, crop, cv::Size(24, 24));
      inputs[i].emplace_back(move(crop));
    }
  }
  vector<cv::String> in_names = { "data1","data2", "data3", "data4", "data5" };
  vector<cv::String> out_names = { "fc5_1", "fc5_2", "fc5_3", "fc5_4", "fc5_5" };
  vector<cv::Mat> out_blobs;
  for (int i = 0; i < 5; i++)
    Lnet.setInput(cv::dnn::blobFromImages(inputs[i], 1.f, cv::Size(), cv::Scalar(), false), in_names[i]);
  Lnet.forward(out_blobs, out_names);
  for (int i = 0; i < 5; i++) {
    const float* data = out_blobs[i].ptr<float>(0);
    for (int j = 0; j < _bboxes.size(); j++) {
      float off_x = data[2 * j + 0] - 0.5f;
      float off_y = data[2 * j + 1] - 0.5f;
      // Dot not make large movement with relative offset > 0.35
      if (fabs(off_x) <= 0.35 && fabs(off_y) <= 0.35) {
        _bboxes[j].fpoints[i] += off_x * patchws[j];
        _bboxes[j].fpoints[i + 5] += off_x * patchws[j];
      }
    }
  }
}
