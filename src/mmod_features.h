/*
 * mmod_features.h
 *
 *  Created on: Sep 8, 2011
 *      Author: Gary Bradski
 */

#ifndef MMOD_FEATURES_H_
#define MMOD_FEATURES_H_
#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <vector>
//serialization
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

namespace boost {
namespace serialization {

  //VERBOSE
  // 1 Routine list, 2 values out, 3 internal values outside of loops, 4 intenral values in loops
  #define FEAT_VERBOSE 0

  #if FEAT_VERBOSE >= 1
  #define FEAT_DEBUG_1(X) do{X}while(false)
  #else
  #define FEAT_DEBUG_1(X) do{}while(false)
  #endif

  #if FEAT_VERBOSE >= 2
  #define FEAT_DEBUG_2(X) do{X}while(false)
  #else
  #define FEAT_DEBUG_2(X) do{}while(false)
  #endif
  #if FEAT_VERBOSE >= 3
  #define FEAT_DEBUG_3(X) do{X}while(false)
  #else
  #define FEAT_DEBUG_3(X) do{}while(false)
  #endif
  #if FEAT_VERBOSE >= 4
  #define FEAT_DEBUG_4(X) do{X}while(false)
  #else
  #define FEAT_DEBUG_4(X) do{}while(false)
  #endif



template<class Archive>
void serialize(Archive & ar, cv::Point &p, const unsigned int version)
{
    ar & p.x & p.y;
}
template<class Archive>
void serialize(Archive & ar, cv::Rect &r, const unsigned int version)
{
    ar & r.x & r.y & r.width & r.height;
}
} // namespace serialization
} // namespace boost

//////////////////////////////////////////////////////////////////////////////////////////////
/**
 *\brief This class stores line mode features for each view and related structures
 *\brief for a specific object for a specified mode
 *
 * This class is mainly called via mmod_objects:: class which contains vector<mmod_features>.
 */
class mmod_features
{
    friend class boost::serialization::access;

public:
	std::string session_ID;              				//For database lookup
	std::string object_ID;								//What's the name of this object
	std::vector<int> frame_number;						//Frame number can be associated with view/pose etc
	std::vector<std::vector<uchar> > features;  		//uchar features, only one bit is on
	std::vector<std::vector<cv::Point> > offsets;   	//the x,y coordinates of each feature
	std::vector<cv::Rect>  bbox;						//bounding box of the features
	std::vector<std::vector<int> > quadUL,quadUR,quadLL,quadLR;//List of features in each quadrant
	cv::Rect max_bounds;								//This rectangle contains the maximum width and and height spanned by all the bbox rectangles

        //---Ivan and Abi's code--- Use Flann to see if speed improves//
        cv::flann::Index flann;

	//---temp--- These were created to optimize feature matching//
	int wstep;											//Flag to convert offsets from cv::Point to uchar*
														//   when set, it is set to the row size of images
	std::vector<std::vector<int> > poff;				//Pointer offsets computed from cv::Point offests above.

	mmod_features();

	mmod_features(std::string &sID, std::string &oID);

	/**
	 * \brief Private function to set the session and object ID
	 * @param sID
	 * @param oID
	 */
	void setup(std::string &sID, std::string &oID);

	/**
	 * \brief Return the number of views stored
	 * @return int number of stored views.
	 */
	int size(){ return((int)(features.size()));};

	//SERIALIZATION
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & session_ID;
        ar & object_ID;
        ar & frame_number;
        ar & features;
        ar & offsets;
        ar & bbox;
        ar & quadUL;
        ar & quadUR;
        ar & quadLL;
        ar & quadLR;
        ar & max_bounds;
        wstep = 0;
    }

	/**
	 * \brief Return the overall bounding rectangle of the vector<Rect>  bbox;
	 *
	 * Return the overall bounding rectangle of the vector<Rect>  bbox; Normally you do not have to call this since it is maintained by the
	 * mmod_general::learn_a_template
	 *
	 * @return Rectangle containing the overall bounding box of the rectangle vector<Rect> bbox;
	 */
	cv::Rect find_max_template_size();


	/**
	 * \brief insert a feature at a given index from an external mmod_features class into this mmod_features class
	 *
	 * We use this when we have learned a new template in mmod_general.learn_a_template, have decided to include it
	 * because no existing template matches it well (found by using mmod_general::display_feature and then
	 * mmod_general::match_a_patch_bruteforce or scalable matching).
	 *
	 * @param f			refererence to mmod_features containing 1 or more features
	 * @param index		the index of which feature we want inserted here from f above
	 * @return			the index into which the indexed value from f was inserted. -1 => error
	 */
	int insert(mmod_features &f, int index);

	/**
	 * \brief  Thus function is called automatically from mmod_general::match_a_patch_bruteforce
	 * \brief  it converts cv::Point offsets into uchar offsets for faster lookup
	 *
	 * This function is purely to optimize matching speed using pre-computed pointer offsets
	 * @param I Any image whose size is the same as currently being used for matching
	 */
	void convertPoint2PointerOffsets(const cv::Mat &I);

        /**
         * \brief This function creates a FLANN index after all templates are learned
         *
         * This function is called to speed up testing later
         * @
         */
        void constructFlannIndex();

        int computeFeatureVecIndex(int width, int height, cv::Point &pt);
};

#endif /* MMOD_FEATURES_H_ */
