/*
 * mmod_mode.h
 *
 *  Created on: Sep 8, 2011
 *      Author: Gary Bradski
 */

#ifndef MMOD_MODE_H_
#define MMOD_MODE_H_
#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <vector>
#include "mmod_general.h"
//SERIALIZATION
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
//VERBOSE
// 1 Routine list, 2 values out, 3 internal values outside of loops, 4 intenral values in loops
#define MODE_VERBOSE 0

#if MODE_VERBOSE >= 1
#define MODE_DEBUG_1(X) do{X}while(false)
#else
#define MODE_DEBUG_1(X) do{}while(false)
#endif

#if MODE_VERBOSE >= 2
#define MODE_DEBUG_2(X) do{X}while(false)
#else
#define MODE_DEBUG_2(X) do{}while(false)
#endif
#if MODE_VERBOSE >= 3
#define MODE_DEBUG_3(X) do{X}while(false)
#else
#define MODE_DEBUG_3(X) do{}while(false)
#endif
#if MODE_VERBOSE >= 4
#define MODE_DEBUG_4(X) do{X}while(false)
#else
#define MODE_DEBUG_4(X) do{}while(false)
#endif

//namespace boost {
//namespace serialization {
//
//template<class Archive>
//void serialize(Archive & ar, cv::Point &p, const unsigned int version)
//{
//    ar & p.x & p.y;
//}
//template<class Archive>
//void serialize(Archive & ar, cv::Rect &r, const unsigned int version)
//{
//    ar & r.x & r.y & r.width & r.height;
//}
//} // namespace serialization
//} // namespace boost


//////////////////////////////////////////////////////////////////////////////////////////////
/**
 *\brief This class stores features for each modality for an object
 *
 * This class has a sorted list (via std::map) of features (for each view)
 * for each modality (depth, gradient ...) for a given object.
 */
class mmod_mode
{
public:
	std::string mode;						//What mode (depth, color, gradient ...) these models learned
	typedef std::map <std::string, mmod_features> ObjectModels; //(object_ID,model) This class consists of objects given a mode
	ObjectModels		objs;			//object model (object_ID, features for each view of the object)
	mmod_general 		util;			//Learning, Matching etc (contains patch w

	//Below is just temp storage for matching convenience when using
//	vector<Rect> rv;			//vector of rectangle bounding boxes from an image match_all_objects
//	vector<float> scores;		//the scores from the above
//	vector<string> ids; 		//the matched object's IDs
//	vector<int> frame_nums;		//the matched object's frame number
	cv::Mat patch; 	//This is a patch for use in display_feature.  This is good for speed, bad for thread safety.
	mmod_mode();

	mmod_mode(const std::string &mode_name);

	/**
	 * \brief Return all the object names in mmod_mode::objs.
	 *
	 * @param obj_names  fill this vector with names
	 * @return	Number of names
	 */
	int return_object_names(std::vector<std::string> &obj_names);

	//SERIALIZATION
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & objs;
    }

    /**
     * \brief Construct FLANN index for each mmod_features contained
     */
    void construct_flann_index();

    /**
	 * \brief Learn a template if no other template matches this view of the object well enough.
	 *
	 * This is the most common method of learning a template: Only actually learn a template if
	 * if no other template is close to the current features in the mask. So, this routine wraps the
	 * other learn_a_template and only actually records the template if no existing template scores above
	 * the learn_thresh.
	 *
	 * @param Ifeat				Feature image of uchar bytes where only one or zero bits are on.
	 * @param Mask				uchar mask of the object to be learned
	 * @param session_ID		Session ID to store if we learn a template
	 * @param object_ID			Object name to store if we learn a template
	 * @param framenum			Frame number of this object, so that we can reconstruct pose from the database
	 * @param learn_thresh		If no features from f match above this, learn a new template.
	 *                          Set to zero to learn all templates (no match search is then done)
	 * @param Score				If set, fill with patch match score
	 * @return					Returns index of newly learned template, or -1 if a template already covered
	 */
	int learn_a_template(cv::Mat &Ifeat, cv::Mat &Mask, std::string &session_ID, std::string &object_ID,
			int framenum, float learn_thresh, float *Score=0);


	/**
	 * \brief Return the match found at a point in the image
	 *
	 * @param object_ID		object name
	 * @param I				Feature image of uchar bytes where only one or zero bits are on.
	 * @param pp			Point to perform the match at
	 * @param match_index	The index of the match will be returned here
	 * @param R				if a match, return boundind box of the feature, else leave alone
	 * @param frame_numb	if a match, return frame_number of feature, else leave alone
	 * @return				Score of this match
	 */
	float match_an_object(std::string &object_ID, const cv::Mat &I, const cv::Point &pp, int &match_index,
			cv::Rect &R, int &frame_numb);

	//	/**
	//	 * \brief Find all objects within the masked part of an image. Do non-maximum suppression on the list
	//	 *
	//	 * Search a whole image within an (optional) mask for objects, skipping (skipY,skipX) each time. Non-max suppress the result.
	//	 * Results are stored in f.rv (rect), f.scores (match value), f.objs (object_IDs) f.frame_nums (frame#s).
	//	 *
	//	 * @param I					Feature image of uchar bytes where only one or zero bits are on.
	//	 * @param Mask				Mask of where to search. If empty, search the whole image. If not empty, it must be CV_8UC1 with same size as I
	//	 * @param match_threshold	Matches have to be above this score [0,1] to be considered a match
	//	 * @param frac_overlap		the fraction of overlap between 2 above threshold feature's bounding box rectangles that constitutes overlap
	//	 * @param skipX				In the search, jump over this many pixels X
	//	 * @param skipY				In the search, jump over this many pixels Y
	//	 * @return					Number of surviving non-max suppressed object matches. Results are stored in f.rv (rect), f.scores (match value), f.objs (object_IDs) f.frame_nums (frame#s).
	//	 */
	//	int match_all_objects(const Mat &I, const Mat &Mask, float match_threshold, float frac_overlap, int skipX = 7, int skipY = 7);



};


#endif /* MMOD_MODE_H_ */
