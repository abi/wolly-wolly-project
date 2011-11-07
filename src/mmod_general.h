/*
 * mmod_general.h
 *
 *  Created on: Sep 8, 2011
 *      Author: Gary Bradski
 */

#ifndef MMOD_GENERAL_H_
#define MMOD_GENERAL_H_
#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <vector>
#include "mmod_features.h"

//DEFINES
#define ORAMT 7   //Amount of ORing to do in each linemod feature image
//VERBOSE
// 1 Routine list, 2 values out, 3 internal values outside of loops, 4 intenral values in loops
#define GENL_VERBOSE 0

#if GENL_VERBOSE >= 1
#define GENL_DEBUG_1(X) do{X}while(false)
#else
#define GENL_DEBUG_1(X) do{}while(false)
#endif

#if GENL_VERBOSE >= 2
#define GENL_DEBUG_2(X) do{X}while(false)
#else
#define GENL_DEBUG_2(X) do{}while(false)
#endif
#if GENL_VERBOSE >= 3
#define GENL_DEBUG_3(X) do{X}while(false)
#else
#define GENL_DEBUG_3(X) do{}while(false)
#endif
#if GENL_VERBOSE >= 4
#define GENL_DEBUG_4(X) do{X}while(false)
#else
#define GENL_DEBUG_4(X) do{}while(false)
#endif
#define FLOATLUT  //Appears that this is faster.
//////////////////////////////////////////////////////////////////////////////////////////////
class mmod_general
{
public:
	std::vector<cv::Mat> acc,acc2;
	int lut[256];//Lookup table converting bit position in a byte (the equivalent number) to its actual bit position
#ifdef FLOATLUT
	std::vector<std::vector<float> > matchLUT; //matchLUT[lut[model_uchar]][image_uchar];
#else
	std::vector<std::vector<int> > matchLUT; //matchLUT[lut[model_uchar]][image_uchar];
#endif
//	std::vector<std::vector<char> > matchLUT; //matchLUT[lut[model_uchar]][image_uchar];

	/**
	 * \brief mmod_general constructor. Fills Cos distances in matchLUT.
	 *
	 * @return
	 */
	mmod_general();


	/**
	 * \brief Take a mmod_features derived uchar feature vector and copy it out to an image structure. This is for matching features to features and for debug.
	 *
	 * Input image I must be single channel uchar and of adequate size to display the whole feature (no bounds checking is done).
	 * The feature is drawn relative to the center pixel so the bounding box should be offset from there. For example, for a 50x60 (width,height) image and
	 * a feature with bounding box of bbwidth 10, bbheight 20, we'd set x = -10/2; and y= -20/2. This easily fits within the image bounds.
	 * To make sure it is of adequate size, just make sure mmod_features::max_bounds.width < I.cols and mmod_features::max_bounds.height < I.rows.
	 *
	 * Example call: display_feature(patch, ftemp.features[index], ftemp.offsets[index], ftemp.bbox[index]); //will draw this feature into "patch"
	 *
	 * @param I		Input image in which to display the feature (must be CV_8UC1 and large enough to contain the feature offsets
	 * @param f		A uchar feature vector
	 * @param o		The offsets from center pixsl of each uchar in the feature vector above
	 * @param bbox  The bounding box of this feature
	 */
	void display_feature(cv::Mat &I, std::vector<uchar> &f, std::vector<cv::Point> &o, cv::Rect &bbox);

	/**
	 * \brief This takes in binarized 8U_C1 image and colorizes it for visualization.
	 *
	 * Colors are Bright red 128, dull red 64, bright yellow 32, dull yellow 16, bright purple 8,
	 * dull purple 4, blue 2, dull blue 1, black 0. I actually like this for visualizing gradients too by color
	 *
	 * @param I		input binarized image [128,64,32,16,8,4,2,1,0] CV_8UC1
	 * @param iB	output colorized image CV_8UC3
	 */
	void visualize_binary_image(const cv::Mat &I, cv::Mat &iB);

	/**
	 * \brief Take an 8UC1 binarized gradient image and visualize it into a color image
	 * visualize_binary_image() will visualize the gradient directions by color -- I prefer that.
	 *
	 * @param ori	Gradient image converted to 8 bit orientations
	 * @param vis	3 Chanel image to draw into
	 * @param skip	Draw gradients every skip pixels, DEFAULT 1
	 * @param R		Rectangular region of interest. DEFAULT: View whole image.
	 */
	void visualize_gradient_orientations(cv::Mat &ori, cv::Mat &vis, int skip = 1, cv::Rect R=cv::Rect(0,0,0,0));

	/**
	 * \brief Draw a feature (vector<uchar> with offsets (vector<Point>) into an image at a point. Does bounds checking
	 *
	 * This function is used for debug (testing matching)
	 *
	 * @param I		Input image, should be CV_8UC1
	 * @param p		Point to draw feature relative to
	 * @param f		Vector of feature values
	 * @param o		Vector of offsets
	 */
	void display_feature_at_Point(cv::Mat &I, cv::Point p, std::vector<uchar> &f, std::vector<cv::Point> &o);

	/**
	 * \brief Brute force match linemod templates at (centered on) a particular point in an image
	 *
	 * This is mainly called via mmod_objects::match_all_objects
	 *
	 * Do a brute force match of all the templates in a given mmod_features at a given pixel (Point) in an image.  It does bounds checking for you.
	 *
	 * @param I				Input image or patch
	 * @param p				Point(x,y) at which to match
	 * @param f				trained mmod_features reference to match against
	 * @param match_index   which feature had the maximal match score
	 * @return				score of maximal match. If f is empty, return 0 (nothing matches)
	 */
	float match_a_patch_bruteforce(const cv::Mat &I, const cv::Point &p, mmod_features &f, int &match_index);


	/**
	 * \brief Brute force match a linemod filter template at (centered on) a particular point in an image
	 *
	 * This is the matching function for mmod_filters to return a score of a given object/modality and given view at a point
	 *
	 * @param I				Input image or patch
	 * @param R				Rectangle at which to match
	 * @param f				trained mmod_features reference to match against
	 * @param index   		index of which view to use in mmod_features
	 * @return				score of match. If f is empty, return 0 (nothing matches)
	 */
	float match_one_feature(const cv::Mat &I, const cv::Rect &R, mmod_features &f, int index);

	/**
	 * Given an 8UC1 image where each pixel is a byte with at most 1 bit on, Either:
	 * 0 OR into each pixel the spanXspan values surrounding that pixel, Or
	 * 1 calculate the bit in the majority within each span x span window and output that "cleaned up" image
	 *
	 *  void SumAroundEachPixel8UC1(Mat &co, Mat &out, int span)
	 *  co  -- input 8UC1 image where each pixel is a byte with at most 1 bit on
	 *  out -- output "cleaned up" image (can be the same as co and is faster that way)
	 *  span -- the size of the spanXspan window in which to calulate the majority
	 *  Or0_Max1 -- If 0, compute the span x span OR, else compute the Majority bit type in a span x span window.
	 */
	void SumAroundEachPixel8UC1(cv::Mat &co, cv::Mat &out, int span = 8, int Or0_Max1 = 0);



	/**
	 *\brief fillCosDist() -- fill up the match lookup table with COS distance functions
	 *
	 * This just fills up matchLUT[9][256] with single bit set byte to byte Cos match look up
	 * for model (single bit set byte) m, and image uchar byte b, the match would be looked up as
	 * matchLUT[lut[m]][b]
	 */
	void fillCosDist();



	/**
	 * \brief Match a uchar to a uchar using the match look up table
	 *
	 * This function is really just an example of how matchLUT is called inner loop.
	 *
	 * @param model_uchar  From the model (a single bit is set only)
	 * @param image_uchar  From the feature image
	 * @return 			Match value
	 */
#ifdef FLOATLUT
	float match(uchar &model_uchar, uchar &image_uchar);
#else
	int match(uchar &model_uchar, uchar &image_uchar);
#endif

	/**
	 * \brief Suppress overlapping rectangle to be the rectangle with the highest score
	 *
	 *
	 * @param rv			vector of rectangle to check
	 * @param scores		their match scores (keep the largest score preferentially)
	 * @param object_ID		class name associated with the rectangle(s)
	 * @param frame_number	frame_number number associated with the rectangles
	 * @param feature_indices This is a vector of features for each mode. mode[].vector<int>. Look up is then modes[modality].objs[name].features[match_index]
	 * @param frac_overlap	the fraction of overlap between 2 rectangles that constitutes overlap
	 * @return 				Num of rectangles cleaned of overlap left in rv.
	 */
	int  nonMaxRectSuppress(std::vector<cv::Rect> &rv, std::vector<float> &scores, std::vector<std::string> &object_ID, std::vector<int> &frame_number,
			std::vector<std::vector<int> > &feature_indices, float frac_overlap);

	/**
	 * \brief Given a binarized feature image and a mask of where to collect features, learn a template there (no matter if other templates match it well).
	 *
	 * This is mainly called via mmod_objects::learn_a_template.
	 *
	 * We often use this by passing in a temporary features class with no entries.  We learn the feature and decide
	 * to include it in our real mmod_features only if no existing template matches it well
	 * (found by using mmod_general::display_feature and then mmod_general::match_a_patch_bruteforce or by using
	 * scalable matching).
	 *
	 * @param Ifeatures   Input image 8U_C1 of binarized features
	 * @param Mask		  Mask 8U_C1 of where the object is
	 * @param framenum	  frame number of this view
	 * @param features	  this will hold our learned template
	 * @return index of template learned in features variable.
	 */
	int learn_a_template(cv::Mat &Ifeatures,  cv::Mat &Mask, int framenum, mmod_features &features );

	/**
	 * \brief Score the current scene's recognition results assuming only one type of trained object in the scene
	 * @param rv				Bounding rectangle of proposed object recognitions
	 * @param ids				Names of the proposed recognized objects
	 * @param currentObj		Name of trained object
	 * @param Mask				Where the object(s) is(are)
	 * @param true_positives	Number of correct detections
	 * @param false_positives	Number of false identifications
	 * @param wrong_object		Number of wrongly classified objects of the trained type
	 * @param R					Rectangle computed from bounding box of Mask on pixels
	 * @return					true_positives, -1 error
	 */
	int score_with_ground_truth(const std::vector<cv::Rect> &rv, const std::vector<std::string> &ids,
			std::string &currentObj, cv::Mat &Mask, int &true_positives,
			int &false_positives, int &wrong_object, cv::Rect &R);

}; //end mmod_general

#endif /* MMOD_GENERAL_H_ */
