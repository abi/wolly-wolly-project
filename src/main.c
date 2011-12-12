#include <opencv2/opencv.hpp>

#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include <stdio.h>

int main(int argc, char** argv) {
  flann::Matrix<float> dataset;
  flann::Matrix<float> query;
  
  cv::flann::Index(NULL, NULL);
  return 0;
}
