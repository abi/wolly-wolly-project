/*
 * mmod_color.cpp
 *
 * Compute line mod color features
 *
 *  Created on: Sep 13, 2011
 *      Author: Gary Bradski
 */
#include "mmod_color.h"
#include <iostream>
#include <stdexcept>
#include <utility>
using namespace cv;
using namespace std;

////////////COLOR HLS///////////////////////////////////////////////////////////
/**
 * \brief Compute a color linemod feature based on Hue values near gradients
 * @param Iin  Input BGR image CV_8UC3
 * @param Icolorord Result image CV_8UC1
 * @param Mask  compute on masked region (can be left empty) CV_8UC3 or CV_8UC1 ok
 * @param mode  If "test", noise reduce and blur the resulting image, "none": do nothing, else noise reduce for training
 */
void colorhls::computeColorHLS(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat &Mask, std::string mode)
{
	//CHECK INPUTS
	CALCFEAT_DEBUG_1(cout << "In colorhls::computeColorHLS" << end;);
	if(Itmp.empty() || Iin.rows != Itmp.rows || Iin.cols != Itmp.cols)
	{
		Itmp.create(Iin.size(),CV_8UC3);
	}
	if(Icolorord.empty() || Iin.rows != Icolorord.rows || Iin.cols != Icolorord.cols)
	{
		Icolorord.create(Iin.size(),CV_8UC1);
	}
	Icolorord = Scalar::all(0); //else make sure it's zero
	cv::Mat temp;
	if (!Mask.empty()) //We have a mask
	{
		if (Iin.size() != Mask.size())
		{
			cerr << "ERROR: Mask in computeColorOrder size != Iina" << endl;
			return;
		}
		if (Mask.type() == CV_8UC3)
		{
			//don't write into the Mask, as its supposed to be const.
			cv::cvtColor(Mask, temp, CV_RGB2GRAY);
		}
		else
			temp = Mask;
	}
	//GET HUE
	cvtColor(Iin, Itmp, CV_BGR2HLS);
	vector<Mat> HLS;
	split(Itmp, HLS);
	double minVal = 0,maxVal = 0;
	CALCFEAT_DEBUG_3(
		cout << "HLS size: " << HLS.size() << endl;
		minMaxLoc(HLS[0], &minVal, &maxVal);
		cout << "HLS0 min = " << minVal << ", HLS max = " << maxVal << endl;
		minMaxLoc(HLS[1], &minVal, &maxVal);
		cout << "HLS1 min = " << minVal << ", HLS max = " << maxVal << endl;
		minMaxLoc(HLS[2], &minVal, &maxVal);
		cout << "HLS2 min = " << minVal << ", HLS max = " << maxVal << endl;
	);

	//ONLY REGISTER HUE AROUND STRONG GRADIENTS
	Scharr( HLS[1], grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT ); //dx
	convertScaleAbs( grad_x, abs_grad_x );

	Scharr( HLS[1], grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT ); //dy
	convertScaleAbs( grad_y, abs_grad_y );
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );	//|dx| + |dy|
	//Set what "strong" gradient means
	Scalar mean,std;
	meanStdDev(grad, mean, std);
	uchar thresh = (uchar)(mean[0] + std[0]*1.1);
	CALCFEAT_DEBUG_3(
		cout << ", mean = " << mean[0] << ", std = " << std[0] << endl;
		cout << "thresh = " << (int)thresh << endl;
		minMaxLoc(grad, &minVal, &maxVal);
		cout << "grad min = " << minVal << ", grad max = " << maxVal << endl;
		);
	//PRODUCE THE COLOR LINE MOD FEATURE IMAGE
	Mat_<uchar>::iterator h = HLS[0].begin<uchar>(),he = HLS[0].end<uchar>();
	Mat_<uchar>::iterator c = Icolorord.begin<uchar>();
	Mat_<uchar>::iterator g = grad.begin<uchar>();
	if(Mask.empty()) //if no mask
	{
		CALCFEAT_DEBUG_3(cout << "No mask" << endl;);
		for(int i = 0; h != he;++h,++c,++g,++i)
		{
			if((*g) > thresh)	//We only compute colors where gradients are large enough
			{
				int rshift = (int)((float)(*h)/22.5); //Break Hue [0,179] into 8 parts
				*c = (uchar)(1<<rshift);              //Convert this to a single bit
				CALCFEAT_DEBUG_3(*h = (uchar)(rshift*18 + 128););
			}
			else
			{
				*c = 0;
				CALCFEAT_DEBUG_3(*h = 0;);
			}
//			if(*c != 0&&*c != 1&&*c != 2&&*c != 4&&*c!=8&&*c!=16&&*c!=32&&*c!=64&&*c!=128)
//				cout << i<<": bad value of c:"<<(int)*c<<endl;
		}
	} else { //If mask
		Mat_<uchar>::const_iterator m = temp.begin<uchar>();
		CALCFEAT_DEBUG_3(cout << "Use mask" << endl;);
		for(int i = 0; h != he;++h,++c,++g,++m,++i)
		{
			if(!(*m)) continue;  //Only compute pixels with corresponding
			if((*g) > thresh)	 //We only compute colors where gradients are large enough
			{
				int rshift = (int)((float)(*h)/22.5);//Break Hue [0,179] into 8 parts
				*c = (uchar)(1<<rshift);             //Convert this to a single bit
				CALCFEAT_DEBUG_3(*h = (uchar)(rshift*18 + 128););
			}
			else
			{
				*c = 0;
				CALCFEAT_DEBUG_3(*h = 0;);
			}
//			if(*c != 0&&*c != 1&&*c != 2&&*c != 4&&*c!=8&&*c!=16&&*c!=32&&*c!=64&&*c!=128)
//				cout << i<<": bad value of c:"<<(int)*c<<endl;
		}
	}
	mmod_general gen;
	if(mode != "none")
		gen.SumAroundEachPixel8UC1(Icolorord,Icolorord,3,1); //Clean the features of spurious gradients
	if(mode == "test")
		gen.SumAroundEachPixel8UC1(Icolorord,Icolorord,ORAMT,0); //Spread features by ORing

	CALCFEAT_DEBUG_3(
		namedWindow("H",0);
		namedWindow("L",0);
		namedWindow("S",0);
		imshow("H",HLS[0]);
		imshow("L",HLS[1]);
		imshow("S",HLS[2]);
		waitKey();
		destroyWindow("H");
		destroyWindow("L");
		destroyWindow("S");
	);
	CALCFEAT_DEBUG_2(cout << "Exit colorhls::computeColorHLS" << end;);
}




////////////////////////GRADIENT FEATURES//////////////////////////////////////////////
/**
 * \brief Compute gradient linemod features from the maximum color plane gradient. Ignores weak gradients
 * @param Iin			Input BGR, CV_8UC3 image
 * @param Icolorord		Output CV_8UC1 image
 * @param Mask			compute on masked region (can be left empty) CV_8UC3 or CV_8UC1 ok
 * @param mode  If "test", noise reduce and blur the resulting image, "none": do nothing, else noise reduce for training
 */
void gradients::computeGradients(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask, std::string mode)
{
	//CHECK INPUTS
	CALCFEAT_DEBUG_1(cout << "In colorhls::computeColorHLS" << end;);
//	if(Itmp.empty() || Iin.rows != Itmp.rows || Iin.cols != Itmp.cols)
//	{
//		Itmp.create(Iin.size(),CV_8UC1);
//	}
	if(Icolorord.empty() || Iin.rows != Icolorord.rows || Iin.cols != Icolorord.cols)
	{
		Icolorord.create(Iin.size(),CV_8UC1);
	}
	Icolorord = Scalar::all(0); //else make sure it's zero
	cv::Mat temp;
	if (!Mask.empty()) //We have a mask
	{
		if (Iin.size() != Mask.size())
		{
			cerr << "ERROR: Mask in computeColorOrder size != Iina" << endl;
			return;
		}
		if (Mask.type() == CV_8UC3)
		{
			//don't write into the Mask, as its supposed to be const.
			cv::cvtColor(Mask, temp, CV_RGB2GRAY);
		}
		else
			temp = Mask;
	}
	//FIND THE MAX GRADIENT RESPONSE ACROSS COLORS
//	cvtColor(Iin, Itmp, CV_RGB2GRAY);
	vector<Mat> RGB;
	split(Iin, RGB);

	Scharr( RGB[0], grad_x, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT ); //dx
	Scharr( RGB[0], grad_y, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT ); //dy
	cartToPolar(grad_x, grad_y, mag0, phase0, true); //True => in degrees not radians
	Scharr( RGB[1], grad_x, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT ); //dx
	Scharr( RGB[1], grad_y, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT ); //dy
	cartToPolar(grad_x, grad_y, mag1, phase1, true); //True => in degrees not radians
	Scharr( RGB[2], grad_x, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT ); //dx
	Scharr( RGB[2], grad_y, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT ); //dy
	cartToPolar(grad_x, grad_y, mag2, phase2, true); //True => in degrees not radians

	//COMPUTE RESONABLE THRESHOLDS
	Scalar mean0,std0,mean1,std1,mean2,std2;
	meanStdDev(mag0, mean0, std0);
	meanStdDev(mag1, mean1, std1);
	meanStdDev(mag2, mean2, std2);
#define stdmul 0.1
	float thresh0 = (float)(mean0[0] + std0[0]*stdmul);///1.25);
	float thresh1 = (float)(mean1[0] + std1[0]*stdmul);///1.25);
	float thresh2 = (float)(mean2[0] + std2[0]*stdmul);///1.25);
	double minVal0,maxVal0,minVal1,maxVal1,minVal2,maxVal2;
	CALCFEAT_DEBUG_3(
		cout <<"     means(B,G,R) ("<<mean0[0]<<", "<<mean1[0]<<", "<<mean2[0]<<")"<<endl;
		cout <<"     std(B,G,R)   ("<<std0[0]<<", "<<std1[0]<<", "<<std2[0]<<")"<<endl;
		minMaxLoc(mag0, &minVal0, &maxVal0);
		minMaxLoc(mag1, &minVal1, &maxVal1);
		minMaxLoc(mag2, &minVal2, &maxVal2);
		cout <<"        minVals(B,G,R) ("<<minVal0<<", "<<minVal1<<", "<<minVal2<<")"<<endl;
		cout <<"        maxVals(B,G,R) ("<<maxVal0<<", "<<maxVal1<<", "<<maxVal2<<")"<<endl;
		cout <<"       thresh(B,G,R) ("<<thresh0<<", "<<thresh1<<", "<<thresh2<<")"<<endl;
	);

	//CREATE BINARIZED OUTPUT IMAGE
	MatIterator_<float> mit0 = mag0.begin<float>(), mit_end = mag0.end<float>();
	MatIterator_<float> pit0 = phase0.begin<float>();
	MatIterator_<float> mit1 = mag1.begin<float>();
	MatIterator_<float> pit1 = phase1.begin<float>();
	MatIterator_<float> mit2 = mag2.begin<float>();
	MatIterator_<float> pit2 = phase2.begin<float>();
	MatIterator_<uchar> bit = Icolorord.begin<uchar>();
	float angle;
	if(Mask.empty()) //if no mask
	{
		for(; mit0 != mit_end; ++mit0, ++pit0,++mit1, ++pit1, ++mit2, ++pit2, ++bit)
		{
			if(*mit0 > *mit1)
			{
				if(*mit0 > *mit2) //mit0 is max
				{
					if(*mit0 < thresh0) continue; //Ignore small gradients
					angle = *pit0;
				}
				else // mit2 is max
				{
					if(*mit2 < thresh2) continue;  //Ignore small gradients
					angle = *pit2;
				}
			}
			else if(*mit1 > *mit2)//mit1 is max
			{
				if(*mit1 > *mit2) //mit1 is max
				{
					if(*mit1 < thresh1) continue; //Ignore small gradients
					angle = *pit1;
				}
				else //mit2 is max
				{
					if(*mit2 < thresh2) continue;  //Ignore small gradients
					angle = *pit2;
				}
			}
			if(angle >= 180.0) angle -= 180.0; //We ignore polarity of the angle
			*bit = 1 << (int)(angle*0.044444444); //This is the floor of angle/(180.0/8) to put the angle into one of 8 bits. Set that bit
		}
	}
	else //There is a mask
	{
		Mat_<uchar>::const_iterator m = temp.begin<uchar>();
		for(; mit0 != mit_end; ++mit0, ++pit0,++mit1, ++pit1, ++mit2, ++pit2, ++bit, ++m)
		{
			if(!(*m)) continue;  //Only compute pixels with corresponding mask pixel set
			if(*mit0 > *mit1)
			{
				if(*mit0 > *mit2) //mit0 is max
				{
					if(*mit0 < thresh0) continue; //Ignore small gradients
					angle = *pit0;
				}
				else // mit2 is max
				{
					if(*mit2 < thresh2) continue;  //Ignore small gradients
					angle = *pit2;
				}
			}
			else if(*mit1 > *mit2)//mit1 is max
			{
				if(*mit1 > *mit2) //mit1 is max
				{
					if(*mit1 < thresh1) continue; //Ignore small gradients
					angle = *pit1;
				}
				else //mit2 is max
				{
					if(*mit2 < thresh2) continue;  //Ignore small gradients
					angle = *pit2;
				}
			}
			if(angle >= 180.0) angle -= 180.0; //We ignore polarity of the angle
			*bit = 1 << (int)(angle*0.044444444); //This is the floor of angle/(180.0/8) to put the angle into one of 8 bits. Set that bit
		}
	}
	//Output feature adjustments
	mmod_general g;
	if(mode != "none")
		g.SumAroundEachPixel8UC1(Icolorord,Icolorord,3,1); //Clean the features of spurious gradients
	if(mode == "test")
		g.SumAroundEachPixel8UC1(Icolorord,Icolorord,ORAMT,0); //Spread features by ORing
}


////////////////////////DEPTH FEATURES//////////////////////////////////////////////
/**
 * \brief Compute linemod features from the gradient of the depth image (CV_16UC1)
 * @param Iin			Input depth image, CV_16UC1
 * @param Icolorord		Output CV_8UC1 of gradient depth features
 * @param Mask			compute on masked region (can be left empty) CV_8UC3 or CV_8UC1 ok
 * @param mode  		If "test", noise reduce and blur the resulting image (DEFAULT), "none": do nothing, else noise reduce for training
 */
void depthgrad::computeDepthGradients(const cv::Mat &Iin, cv::Mat &Icolorord, const cv::Mat Mask, std::string mode)
{
	//THIS IS NOT TESTED YET.  FOR INSTANCE, I MIGHT DECIDE NOT TO USE THRESHOLDS BELOW AT ALL
	//CHECK INPUTS
	CALCFEAT_DEBUG_1(cout << "In depthgrad::computeDepthGradients" << end;);
//	if(Itmp.empty() || Iin.rows != Itmp.rows || Iin.cols != Itmp.cols)
//	{
//		Itmp.create(Iin.size(),CV_8UC1);
//	}
	if(Iin.type() != CV_16UC1)
	{
		cerr << "ERROR: Depth image is not of type CV_16UC1" << endl;
		return;
	}
	if(Icolorord.empty() || Iin.rows != Icolorord.rows || Iin.cols != Icolorord.cols)
	{
		Icolorord.create(Iin.size(),CV_8UC1);
	}
	Icolorord = Scalar::all(0); //else make sure it's zero
	cv::Mat temp;
	if (!Mask.empty()) //We have a mask
	{
		if (Iin.size() != Mask.size())
		{
			cerr << "ERROR: Mask in computeColorOrder size != Iina" << endl;
			return;
		}
		if (Mask.type() == CV_8UC3)
		{
			//don't write into the Mask, as its supposed to be const.
			cv::cvtColor(Mask, temp, CV_RGB2GRAY);
		}
		else
			temp = Mask;
	}
	//FIND THE MAX GRADIENT RESPONSE ACROSS COLORS
	Scharr( Iin, grad_x, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT ); //dx
	Scharr( Iin, grad_y, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT ); //dy
	cartToPolar(grad_x, grad_y, mag0, phase0, true); //True => in degrees not radians

	//COMPUTE RESONABLE THRESHOLDS
	Scalar mean0,std0;
	meanStdDev(mag0, mean0, std0);
#define stdmul2 0.1
	float thresh0 = (float)(mean0[0] + std0[0]*stdmul2);
	//CREATE BINARIZED OUTPUT IMAGE
	MatIterator_<float> mit0 = mag0.begin<float>(), mit_end = mag0.end<float>();
	MatIterator_<float> pit0 = phase0.begin<float>();
	MatIterator_<uchar> bit = Icolorord.begin<uchar>();
	if(Mask.empty()) //if no mask
	{
		for(; mit0 != mit_end; ++mit0, ++pit0, ++bit)
		{
			if(*mit0 < thresh0) continue; //Ignore small gradients
			*bit = 1 << (int)((*pit0)*0.022222222); //This is the floor of angle/(360.0/8) to put the angle into one of 8 bits. Set that bit
		}
	}
	else //There is a mask
	{
		Mat_<uchar>::const_iterator m = temp.begin<uchar>();
		for(; mit0 != mit_end; ++mit0, ++pit0, ++bit, ++m)
		{
			if(!(*m)) continue;  //Only compute pixels with corresponding mask pixel set
			if(*mit0 < thresh0) continue; //Ignore small gradients
			*bit = 1 << (int)((*pit0)*0.022222222); //This is the floor of angle/(360.0/8) to put the angle into one of 8 bits. Set that bit
		}
	}
	//Output feature adjustments
	mmod_general g;
	if(mode != "none")
		g.SumAroundEachPixel8UC1(Icolorord,Icolorord,3,1); //Clean the features of spurious gradients
	if(mode == "test")
		g.SumAroundEachPixel8UC1(Icolorord,Icolorord,ORAMT,0); //Spread features by ORing
}


