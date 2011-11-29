/*
 * mmod_color.h
 *
 * This file really should be named "Compute Binarized Features" or some such because these are classes that
 * implement turning a modality into a binary image of linemode features.
 *
 *  Created on: Sep 13, 2011
 *      Author: Gary Bradski
 */

#ifndef MMOD_COLOR_H_
#define MMOD_COLOR_H_
#include <opencv2/opencv.hpp>
#include "mmod_general.h"

#define CALCFEAT_SHOW() {std::cout << __FILE__ << " : "  << __LINE__ << std::endl;}
  //VERBOSE
  // 1 Routine list, 2 values out, 3 internal values outside of loops, 4 intenral values in loops
  #define CALCFEAT_VERBOSE 4
    
  #if CALCFEAT_VERBOSE >= 1
  #define CALCFEAT_DEBUG_1(X) do{CALCFEAT_SHOW() X }while(false)
  #else
  #define CALCFEAT_DEBUG_1(X) do{}while(false)
  #endif

  #if CALCFEAT_VERBOSE >= 2
  #define CALCFEAT_DEBUG_2(X) do{CALCFEAT_SHOW() X}while(false)
  #else
  #define CALCFEAT_DEBUG_2(X) do{}while(false)
  #endif
  #if CALCFEAT_VERBOSE >= 3
  #define CALCFEAT_DEBUG_3(X) do{X}while(false)
  #else
  #define CALCFEAT_DEBUG_3(X) do{}while(false)
  #endif
  #if CALCFEAT_VERBOSE >= 4
  #define CALCFEAT_DEBUG_4(X) do{CALCFEAT_SHOW() X}while(false)
  #else
  #define CALCFEAT_DEBUG_4(X) do{}while(false)
  #endif



////////////COLOR HLS///////////////////////////////////////////////////////////
class colorhls {
	cv::Mat Itmp;
	cv::Mat grad_x, grad_y, grad;
	cv::Mat abs_grad_x, abs_grad_y;
public:
	/**
	 * \brief Compute a color linemod feature based on Hue values near gradients
	 * @param Iin  Input BGR image CV_8UC3
	 * @param Icolorord Result image CV_8UC1
	 * @param Mask  compute on masked region (can be left empty) CV_8UC3 or CV_8UC1 ok
	 * @param mode  If "test", noise reduce and blur the resulting image (DEFAULT), "none": do nothing, else noise reduce for training
	 */
	void computeColorHLS(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat &Mask, std::string mode = "test");
};

////////////Gradients///////////////////////////////////////////////////////////
class gradients {
	cv::Mat mag0, phase0, mag1, phase1, mag2, phase2;			//Temp store for gradient processing
	cv::Mat grad_x, grad_y;
	std::vector<cv::Mat> RGB;									//Just temp store split
public:

	////////////////////////GRADIENT FEATURES//////////////////////////////////////////////
	/**
	 * \brief Compute gradient linemod features from the maximum color plane gradient. Ignores weak gradients
	 * @param Iin			Input BGR, CV_8UC3 image
	 * @param Icolorord		Output CV_8UC1 image
	 * @param Mask			compute on masked region (can be left empty) CV_8UC3 or CV_8UC1 ok
	 * @param mode  If "test", noise reduce and blur the resulting image (DEFAULT), "none": do nothing, else noise reduce for training
	 */
	void computeGradients(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask, std::string mode = "test" );

};


////////////Depth Gradients/////////////////////////////////////////////////////
class depthgrad {
	cv::Mat mag0, phase0;			//Temp store for gradient processing
	cv::Mat grad_x, grad_y;

public:

	////////////////////////DEPTH FEATURES//////////////////////////////////////////////
	//THIS IS NOT TESTED YET.  FOR INSTANCE, I MIGHT DECIDE NOT TO USE THRESHOLDS BELOW AT ALL
	/**
	 * \brief Compute linemod features from the gradient of the depth image (CV_16UC1)
	 * @param Iin			Input depth image, CV_16UC1
	 * @param Icolorord		Output CV_8UC1 of gradient depth features
	 * @param Mask			compute on masked region (can be left empty) CV_8UC3 or CV_8UC1 ok
	 * @param mode  		If "test", noise reduce and blur the resulting image (DEFAULT), "none": do nothing, else noise reduce for training
	 */
	void computeDepthGradients(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask, std::string mode = "test");


};


#endif /* MMOD_COLOR_H_ */
