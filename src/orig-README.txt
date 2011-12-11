This is Gary Bradski's 10/19/11 multi-modal perception/version of linemod.  This is just a basic guide to use, it won't compile.
Below, I use the a color image to compute gradient linemod features and the same color image to compute the color features for filtering.

//MAIN FUNCTION (EXAMPLE WITH LOTS OF DEBUG IN IT)
My Main file is in MyMMod.cpp

//CLASSES ARE:
mmod_objects -- holds recognition results, it contains a map of modalities (color, depth, gradients ...) in
mmod_mode	 -- holds a features for a modality, it contains a map of features and offsets which are in
mmod_features -- holds a list of object views (vector<vector<uchar> >) of 8 bit binarized features and their offsets ... these are the model templates

mmod_general  -- Almost all the learning and matching computation and utility functions are here
mmod_color    -- Shouldn't be named "color", should be named mmod_calc_feature -- these classes, one for each feature take a modality as input 
                 (depth image, color image) and creates a feature image of 8 bit values. These take a mask (training) or not (test), see below.

//////////////////A WALK THROUGH OF HOW TO CALL THESE FUNCTIONS//////////////////
//INCLUDES
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <valgrind/callgrind.h>

#include "mmod_general.h"
#include "mmod_objects.h"
#include "mmod_mode.h"
#include "mmod_features.h"
#include "mmod_color.h"

//For serialization
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace cv;
using namespace std;

//INSTANTIATE STUFF:
	mmod_objects Objs; 				//Object train and test.
	mmod_general g;    				//General utilities
	mmod_filters filt("Color"); 	//For post recognition filter object check, I'm using color, you can use any (one) mode
	colorhls calcHLS;				//Color feature processing
	gradients calcGrad;    			//Gradient feature processing
//	depthgrad  calcDepth;			//Depth feature processing
	Mat colorfeat, depthfeat, gradfeat;  //To hold feature outputs. These will be CV_8UC1 images
	Mat ColorRaw0,Mask0,ColorRaw,Mask,noMask; //Will hold raw  images and masks
	Mat Ivis,Gvis;					//feature images, CV_8UC1
	vector<Mat> ColorPyr,MaskPyr;	//Image pyramid 
	vector<Mat> FeatModes; 			//List of images
	vector<string> modesCD; 		//Names of modes (color and depth)
	string SessionID, ObjectName;   //Fill these in.
	int framenum = 0;
	float Score;

	//SET UP:
	//Set up our modes (right now we have color gradients and depth. Below, I just use gradients
#define PYRLEVELS 2 //How many levels of pyramid reduction
#define SKIPAMT 8	//SKIP Amount (skipX, skipY)
//	modesCD.push_back("Color");
	modesCD.push_back("Grad");
	//		modesCD.push_back("Depth");
	float learn_thresh = 0.97; //Just a guess
	float match_threshold = 0.97; //Total guess
	float frac_overlap = 0.5; //the fraction of overlap between 2 above threshold feature's bounding box rectangles that constitutes "overlap"
	float cthresh = 0.91; //Color thresh

. . .

//GET OUR IMAGES IN, AND REDUCE IF YOU WANT A SPEED UP:
... Mat::ColorRaw0 ... //BGR image of type CV_8UC3
... Mat::DepthRaw0 ... //Depth image of type CV_16UC1
... Mat::Mask0 ...     //Object mask of type CV_8UC1 or CV_8UC3

	buildPyramid(ColorRaw0, ColorPyr, PYRLEVELS); //This is optional, you can just use the raw images. I had 1Kx1K images, so did this for speed.
	buildPyramid(Mask0,MaskPyr,PYRLEVELS);
	ColorRaw = ColorPyr[PYRLEVELS];
	Mask = MaskPyr[PYRLEVELS];


//PROCESS TO GET FEATURES
	calcHLS.computeColorHLS(ColorRaw,colorfeat,Mask,"train");
	calcGrad.computeGradients(ColorRaw,gradfeat,Mask,"train");
//		calcDepth.computeDepthGradients(DepthRaw,depthfeat,Mask,"train");
	FeatModes.clear(); //Stack the features ... here I only use gradients
	FeatModes.push_back(gradfeat);
//		FeatModes.push_back(colorfeat);
//		FeatModes.push_back(depthfeat);


//LEARN A TEMPLATE (for now, it will slow down with each view learned).
	int num_templ = Objs.learn_a_template(FeatModes,modesCD, Mask,
			SessionID, ObjectName, framenum, learn_thresh, &Score);

//LEARN A FILTER TO CONFIRM RECOGNIZED OBJECTS IN TEST MODE
	int num_fs = filt.learn_a_template(colorfeat,Mask,"Tea",framenum);

. . . 

//EXAMPLE OF I/O -- I USE A FILTER IN THIS EXAMPLE, YOU CAN USE OBJECTS IN A SIMILAR WAY
	{ //Serialize out 
		cout << "Writing models filt.txt out" << endl;
		std::ofstream ofs("filt.txt");
		boost::archive::text_oarchive oa(ofs);
		oa << filt;
	}
    // ----
	mmod_filters filt2("foo");
    { //Serialize in 
    	cout << "Reading models filt.txt in" << endl;
    	std::ifstream ifs("filt.txt");
    	boost::archive::text_iarchive ia(ifs);
        // read class state from archive
    	ia >> filt2;
    }

. . . 

//TEST MODE
	int skipX = SKIPAMT, skipY = SKIPAMT;  //These control sparse testing of the feature images, here SKIPAMT 8
. . . ColorRaw0 is read in . . .
	buildPyramid(ColorRaw0, ColorPyr, PYRLEVELS); //PYRAMID DOWN IF YOU WANT (FOR SPEED)
	ColorRaw = ColorPyr[PYRLEVELS];					//OR, could just use the full scale image
	//Calculate features
	calcHLS.computeColorHLS(ColorRaw,colorfeat,noMask,"test");
	calcGrad.computeGradients(ColorRaw,gradfeat,noMask,"test");
//		calcDepth.computeDepthGradients(DepthRaw,depthfeat,noMask,"test");
		FeatModes.clear();
		FeatModes.push_back(gradfeat);  //I'm just using gradients here, you can push back as many modalities as you like
//		FeatModes.push_back(colorfeat);	//Color will instead be used for filtering
		//		FeatModes.push_back(depthfeat);

   //RECOGNIZE
   	int num_matches = Objs.match_all_objects(FeatModes,modesCD,noMask,
			                                 match_threshold,frac_overlap,skipX,skipY,&numrawmatches);
   . . . Optionally, check the recognitions with a filter (here my trained color filter)
	filt.filter_object_recognitions(colorfeat,Objs,cthresh);

//USE THE RESULTS
	Objs.rv  		contains a vector of rectanglular bounding boxes of recognized objects
	Objs.ids 		contains the corrsponsing list of their names
	Objs.scores		contains their matching scores
	Objs.frame_nums	contains the frame number of the model, so that we can relate it to the view learned.                          

//UTILITIES
g.visualize_binary_image(gradfeat,Gvis); //DISPLAY COLOR CODED BINARY IMAGE
Objs.draw_matches(ColorRaw);	  //DISPLAYS RECOGNITION RESULTS ONTO THE RAW IMAGE
                             
