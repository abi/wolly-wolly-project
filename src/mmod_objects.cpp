/*
 * mmod_objects.cpp
 *
 *  Created on: Sep 8, 2011
 *      Author: Gary Bradski
 */
#include "mmod_objects.h"
#include <sstream>

using namespace cv;
using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
/**
 *\brief This class stores multi-mod object models
 */

/**
 * \brief Empty all vectors.
 */
void
mmod_objects::clear_matches()
{
  if (!rv.empty() || !modes_used.empty())
  {
    rv.clear();
    scores.clear();
    ids.clear();
    frame_nums.clear();
    modes_used.clear();
    feature_indices.clear();
  }
}

/**
 *\brief  Draw matches after a call to match_all_objects. This function is for visualization
 * @param I   Image you want to draw onto, must be CV_8UC3. No bounds checking done
 * @param o   Offset for drawing
 */
void
mmod_objects::draw_matches(Mat &I, Point o)
{
  OBJS_DEBUG_1(cout<<"mmod_objects::draw_matches Iw,h("<<I.cols<<","<<I.rows<<") at ("<<o.x<<", "<<o.y<<")"<<endl;);
  int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
  double fontScaleS = 0.2;
  double fontScaleO = 0.3;
  int thickness = 1;
  stringstream ss;

  vector<Rect>::iterator ri; //Rectangle iterator
  vector<float>::iterator si; //Scores iterator
  vector<string>::iterator ii; //Object ID iterator (object names)
  vector<vector<int> >::iterator fitr;//Feature indices iterator
  int len = (int) rv.size();
  if (rv.empty())
    len = 1;
  int Dcolor = 150 / len;
  Scalar color(255, 255, 255);
  string stringscore;
  vector<string>::iterator moditr; //Mode iterator
  int indices;
  int num_modes = (int) modes_used.size();
  OBJS_DEBUG_3(cout << "num_modes: " << num_modes << endl;);
  if (num_modes == 0)
    num_modes = 1;
  int dmode = 150 / num_modes;
  int i;
  OBJS_DEBUG_3(
  cout << "rv.s:"<<rv.size()<<", scores.s:"<<scores.size()<<", ids.s"<<ids.size()<<", fi.s:"<<feature_indices.size()<<endl;
  );
  for (i = 0, ri = rv.begin(), si = scores.begin(), ii = ids.begin(), fitr = feature_indices.begin();
		  ri != rv.end(); ++ri, ++si, ++ii, ++i, ++fitr)
  {
    color[i % 3] -= Dcolor; //Provide changing color
    OBJS_DEBUG_4(cout << "i:" << i << " ri:" << ri->x << "," << ri->y << "," <<ri->width<<","<<ri->height<<endl;);
    Rect R(ri->x + o.x, ri->y + o.y, ri->width, ri->height);
    OBJS_DEBUG_4(cout << "R:"<<R.x<<","<<R.y<<","<<R.width<<","<<R.height<<endl;);
   int cx = R.x + R.width / 2, cy = R.y + R.height / 2;
   OBJS_DEBUG_4(cout << "cx("<<cx<<","<<cy<<")"<<endl;);
    rectangle(I, R, color); //Draw rectangle
    ss << *si; //Number to string
    ss >> stringscore;
    //			cout << "string score = " << stringscore << ", *si = " << *si << endl;
    putText(I, *ii, Point(R.x, R.y - 2), fontFace, fontScaleO, color, thickness, 8); //Object ID, and then score
    putText(I, stringscore, Point(R.x + 1, R.y + R.height / 2), fontFace, fontScaleS, color, thickness, 8);
    //DRAW THE ACTUAL FEATURES THEMSELVES
    OBJS_DEBUG_4(cout << "modes_used.size="<<modes_used.size() << endl;);
    for (indices = 0, moditr = modes_used.begin(); moditr != modes_used.end(); ++moditr, ++indices)
    {
      int matchIdx = (*fitr)[indices];
      OBJS_DEBUG_4(cout << "i:"<<i<<" indices("<<*moditr<<")# " << indices << ", matchIdx = "<< matchIdx << endl;);
       vector<Point>::iterator pitr = modes[*moditr].objs[*ii].offsets[matchIdx].begin();
      for (; pitr != modes[*moditr].objs[*ii].offsets[matchIdx].end(); ++pitr)//, ++ucharitr)
      {
    	int Y = pitr->y + cy, X = pitr->x + cx;
    	if(Y < 0 || Y >= I.rows || X < 0 || X >= I.cols) {continue;}
        I.at<Vec3b> (Y,X)[1] = 255 - indices * dmode; //draw in decreasing color for each mode
      }
    }
  }//End for each surviving non-max suppressed feature left
  OBJS_DEBUG_1(cout << "And out of draw_matches" << endl;);
}

/**
 *\brief  cout all matches after a call to match_all_objects. This function is for debug
 *
 *@return Total number of matches
 */
int
mmod_objects::cout_matches()
{
  vector<Rect>::iterator ri;
  vector<float>::iterator si;
  vector<string>::iterator ii;
  string stringscore;
  int i;
  for (ri = rv.begin(), si = scores.begin(), ii = ids.begin(); ri != rv.end(); ++ri, ++si, ++ii)
  {
    cout << *ii << ": score = " << *si << " at R(" << ri->x << ", " << ri->y << ", " << ri->width << ", " << ri->height
        << ")" << endl;
  }
  return (int)(rv.size());
}

/**
 * \brief Find all objects above match_threshold at a point in image.
 *
 * Search for all matches at a point in an image over the modes (CV_8UC1 feature images) that are passed in.
 * No checking done if vectors are empty
 * Results are stored in rv (rect), scores (match value), objs (object_IDs) frame_nums (frame#s).
 *
 * @param I					For each mode, Feature image of uchar bytes where only one or zero bits are on.
 * @param mode_names		List of names of the modes of the above features
 * @param pp				Point at which to search
 * @param match_threshold	Matches have to be above this score [0,1] to be considered a match
 * @return					Number of object matches above match_threshold Results are stored in
 * 							rv (rect), scores (match value), objs (object_IDs) frame_nums (frame#s),
 * 							modes_used (feature types ... gradient, depth...) and feature_indices (which mmod_feature vec matched).
 */
int
mmod_objects::match_all_objects_at_a_point(const vector<Mat> &I, const vector<string> &mode_names, const Point &pp,
                                           float match_threshold)
{
  clear_matches();
  vector<Mat>::const_iterator Iit;
  vector<string>::const_iterator modit;

  //Collect matches
  vector<string> obj_names; //To be filled by return_object_names below
  vector<string>::iterator nit; //obj_names iterator
  ModelsForModes::iterator mfmit = modes.begin();
  int num_names = mfmit->second.return_object_names(obj_names);

  int match_index, frame_number;
  float score;
  float norm = (float) I.size();
  Rect R;
  //GO THROUGH EACH OBJECT
  vector<int> match_indices;
  bool collect_modes = true;
  for (nit = obj_names.begin(); nit != obj_names.end(); ++nit)
  {
    score = 0.0;
    match_indices.clear();
    //GO THROUGH EACH MODE SUMMING SCORES
    for (modit = mode_names.begin(), Iit = I.begin(); modit != mode_names.end(); ++Iit, ++modit)
    {
      if (modes.count(*modit) > 0) //We have this mode
      {
        if (collect_modes)
          modes_used.push_back(*modit);
        score += modes[*modit].match_an_object(*nit, *Iit, pp, match_index, R, frame_number);
        match_indices.push_back(match_index);
        //					objs_modal_features.push_back(modes[*modit].objs[*nit].features[match_index]);
      }
    }
    collect_modes = false;
    score /= norm; //Normalize by number of modes
    if (score > match_threshold) //If we have a match, enter it as a contender
    {
      rv.push_back(Rect(R.x + R.width / 2, R.y + R.height / 2, R.width, R.height));//Our rects are middle based, make this Upper Left based
      scores.push_back(score);
      ids.push_back(*nit);
      frame_nums.push_back(frame_number);
      feature_indices.push_back(match_indices);
      //				object_feature_map.insert(pair<string, vector<vector<uchar> > >(*nit,objs_modal_features));
    }
  }
  return (int) rv.size();
}

/**
 * \brief Find all objects within the masked part of an image. Do non-maximum suppression on the list
 *
 * Search a whole image within an (optional) mask for objects, skipping (skipY,skipX) each time. Non-max suppress the result.
 * Results are stored in rv (rect), scores (match value), objs (object_IDs) frame_nums (frame#s).
 *
 * @param I					For each mode, Feature image of uchar bytes where only one or zero bits are on.
 * @param mode_names		List of names of the modes of the above features
 * @param Mask				Mask of where to search. If empty, search the whole image. If not empty, it must be CV_8UC1 with same size as I
 * @param match_threshold	Matches have to be above this score [0,1] to be considered a match
 * @param frac_overlap		the fraction of overlap between 2 above threshold feature's bounding box rectangles that constitutes overlap
 * @param skipX				In the search, jump over this many pixels X
 * @param skipY				In the search, jump over this many pixels Y
 * @param rawmatches		If set, fill this with the total number of matches before non-max suppression.
 * @return					Number of surviving non-max suppressed object matches.
 * 							rv (rect), scores (match value), objs (object_IDs) frame_nums (frame#s),
 * 							modes_used (feature types ... gradient, depth...) and feature_indices (which mmod_feature vec matched).
 */
int
mmod_objects::match_all_objects(const vector<Mat> &I, const vector<string> &mode_names, const Mat &Mask,
                                float match_threshold, float frac_overlap, int skipX, int skipY, int *rawmatches)
{
  OBJS_DEBUG_1(
      cout << "mmod_objects::match_all_objects, for modes:"<<endl;
      vector<string>::const_iterator vsi;
      for(vsi=mode_names.begin();vsi != mode_names.end();++vsi)
      {
        cout << *vsi << endl;
      }
      cout << "match_thresh:"<<match_threshold<<" frac_overlap:"<<frac_overlap<< " skipxy="<<skipX<<", "<<skipY<<endl;
  );
  clear_matches();
  if (I.empty())
  {
    cerr << "ERROR, in match_all_objects, feature vector is empty." << endl;
    return -1;
  }
  vector<Mat>::const_iterator Iit;
  vector<string>::const_iterator modit;
  if (!Mask.empty())
  {
    for (Iit = I.begin(), modit = mode_names.begin(); Iit != I.end(); ++Iit, ++modit)
    {
      if (Iit->size() != Mask.size())
      {
        cerr << "ERROR in match_all_objects: I[" << *modit << "].size.width(" << (Iit->size()).width
            << ") != Mask.size(" << (Mask.size()).width << ")" << endl;
        return -1;
      }
      if (Iit->type() != Mask.type())
      {
        cerr << "ERROR in match_all_objects: I[" << *modit << "].type(" << Iit->type() << ") != Mask.type("
            << Mask.type() << ")" << endl;
        return -1;
      }
    }
  }
  //Collect matches
  vector<string> obj_names; //To be filled by return_object_names below
  vector<string>::iterator nit; //obj_names iterator
  ModelsForModes::iterator mfmit = modes.begin();
  int num_names = mfmit->second.return_object_names(obj_names);

  int match_index, frame_number;
  float score;
  float norm = (float) I.size();
  OBJS_DEBUG_3(
		  cout << "In mmod_objects::match_all_objects, norm = " << norm << endl;
  	  	  cout << "rows: " << I[0].rows << ", cols: " << I[0].cols << endl;
  );
  Rect R;
  vector<int> match_indices;
  bool collect_modes = true;
  if (Mask.empty()) //THERE IS NO MASK, SEARCH THE WHOLE IMAGE:
  {
    OBJS_DEBUG_3(cout << "In mask empty" << endl;);
    int reportfactory = 6*skipY;
    int reportfactorx = 20*skipX;
    for (int y = 0; y < I[0].rows; y += skipY)
    {
      for (int x = 0; x < I[0].cols; x += skipX)
      {
        //go through each object,
        for (nit = obj_names.begin(); nit != obj_names.end(); ++nit)
        {
          score = 0.0;
          match_indices.clear();
          //go through each mode, summing scores
          for (modit = mode_names.begin(), Iit = I.begin(); modit != mode_names.end(); ++Iit, ++modit)
          {
            if (modes.count(*modit) > 0) //We have this mode
            {
              if (collect_modes)
              {
            	  modes_used.push_back(*modit);
              }
              Point pp = Point(x, y);
              score += modes[*modit].match_an_object(*nit, *Iit, pp, match_index, R, frame_number);
              match_indices.push_back(match_index);
              OBJS_DEBUG_4(
                  if(!(y%reportfactory)&&(!(x%reportfactorx)))
                    cout <<"match Frm#:"<<frame_number<<" For obj["<<*nit<<"], mode["<<*modit<<"] at point("<<x<<","<<y<<
                    ") R("<<R.x<<","<<R.y<<","<<R.width<<","<<R.height<<"), score acc: " <<
                    score << " match_indx: " << match_index << endl;
              );
//              if(!(y%reportfactory)&&(!(x%reportfactorx)))
//              cout <<"("<<x<<","<<y<<") match Frm#:"<<frame_number<<" For obj["<<*nit<<"], mode["<<*modit<<"] at point("<<x<<","<<y<<
//                  ") R("<<R.x<<","<<R.y<<","<<R.width<<","<<R.height<<"), score acc: " <<
//                  score << " match_indx: " << match_index << endl;
           }
          }
          collect_modes = false;
          score /= norm; //Normalize by number of modes
          OBJS_DEBUG_4(
        		  if(!(y%reportfactory)&&(!(x%reportfactorx))) cout << "norm = "<<norm<<", score = "<<score<<" >? "<< match_threshold<<endl;
          );
          if (score > match_threshold) //If we have a match, enter it as a contender
          {
            rv.push_back(Rect(R.x + x, R.y + y, R.width, R.height));//Our rects are middle based, make this Upper Left based
            OBJS_DEBUG_4(
            		if(!(y%reportfactory)&&(!(x%reportfactorx))) cout << "  R("<<R.x<<","<<R.y<<","<<R.width<<","<<R.height<<")"<<endl;
            );
            scores.push_back(score);
            ids.push_back(*nit);
            frame_nums.push_back(frame_number);
            feature_indices.push_back(match_indices);
          }
       }//end for each obj
      }//end for x
    }//end going over rows of the image
  }//end if mask empty
  else //USE THE MASK:
  {
	  cout<< "WE SHOULDN'T BE USING THE MASK..."<<endl;
    OBJS_DEBUG_3(cout << "obj_names.size() = " << obj_names.size() << ", modes = " << mode_names.size() << endl;
    );
    for (int y = 0; y < I[0].rows; y += skipY)
    {
      const uchar *m = Mask.ptr<uchar> (y);
      for (int x = 0; x < I[0].cols; x += skipX, m += skipX)
      {
        OBJS_DEBUG_4(cout << "(" << x << ", " << y << ") m= " << (int)(*m) << endl;);
        if (*m) //Mask covers this point
        {
          OBJS_DEBUG_4(if(!(y%20) && (x == y)) cout << "x,y(" << x << ", " << y << ")";);
          //go through each object,
          for (nit = obj_names.begin(); nit != obj_names.end(); ++nit)
          {
            OBJS_DEBUG_4(if(!(y%20) && (x == y)) cout << "\nvvvvv"<< *nit <<"vvvvv" << endl;);
            score = 0.0;
            match_indices.clear();
            //go through each mode, summing scores
            for (modit = mode_names.begin(), Iit = I.begin(); modit != mode_names.end(); ++Iit, ++modit)
            {
              if (modes.count(*modit) > 0) //We have this mode
              {
                if (collect_modes)
                {
                  modes_used.push_back(*modit);
                  OBJS_DEBUG_4(cout << "modes_used.push_back("<<*modit<<"), len"<<modes_used.size()<<endl;);
                }
                Point pp = Point(x, y);
                float tscore;
                tscore = modes[*modit].match_an_object(*nit, *Iit, pp, match_index, R, frame_number);
                match_indices.push_back(match_index);
                score += tscore;
                OBJS_DEBUG_4(if(!(y%20) && (x == y)) cout << "For object " << *nit << ", mode " << *modit << ",score = " << tscore <<
                    ", cumscore = " << score << " R(" << R.x <<","<<R.y<<","<<R.width<<","<<R.height<<")"<< endl;);
              }
            }
            collect_modes = false;
            score /= norm; //Normalize by number of modes
            OBJS_DEBUG_4(if(!(y%20) && (x == y)) cout << "score(" << score << ") >? match_threshold = " << match_threshold << endl;);
            if (score > match_threshold) //If we have a match, enter it as a contender
            {
              OBJS_DEBUG_4(if(!(y%10) && (x == y)) cout << *nit << " Push back R(" << R.x + x<< ", " << R.y + y<< ", " << R.width << ", " << R.height << ")" << endl;);
              rv.push_back(Rect(R.x + x, R.y + y, R.width, R.height)); //Our rects are middle based, make this Upper Left based
              scores.push_back(score);
              ids.push_back(*nit);
              frame_nums.push_back(frame_number);
              feature_indices.push_back(match_indices);
            }
          }
        }
      }
    }//end going over rows of the image
  }
  OBJS_DEBUG_3(cout << "Pre nonMax, we have " << rv.size() << " potential objects" << endl;);

  //Get rid of spurious overlaps:
  if(rawmatches)
	  *rawmatches = (int)(rv.size());
  int num_objs = util.nonMaxRectSuppress(rv, scores, ids, frame_nums, feature_indices, frac_overlap);

  OBJS_DEBUG_2(cout << "Post nonMax, we have " << rv.size() << " potential objects" << endl;
      cout << "____________________\n" << endl;);
//  cout << "num_objs " << num_objs << endl;
  return num_objs;
}

/**
 * \brief Learn a template if no other template matches this view of the object well enough.
 *
 * This is the most common method of learning a template: Only actually learn a template if
 * if no other template is close to the current features in the mask. So, this routine wraps the
 * other learn_a_template and only actually records the template if no existing template scores above
 * the learn_thresh.
 *
 * @param Ifeats			For each modality, Feature image of uchar bytes where only one or zero bits are on.
 * @param mode_names		List of names of the modes of the above features
 * @param Mask				uchar mask of the object to be learned
 * @param framenum			Frame number of this object, so that we can reconstruct pose from the database
 * @param learn_thresh		If no features from f match above this, learn a new template.
 * @param Score				If set, fill with patch match score
 * @return					Returns total number of templates for this object
 */
int mmod_objects::learn_a_template(vector<Mat> &Ifeat, const vector<string> &mode_names, Mat &Mask, string &session_ID,
                               string &object_ID, int framenum, float learn_thresh, float *Score)
{
  OBJS_DEBUG_1(
      cout << "mmod_objects::learn_a_template(sesID:"<<session_ID<<", objID:"<<object_ID<<" frame#:"<<framenum
           <<" learn_thresh:"<<learn_thresh<<endl;
	  vector<string>::const_iterator vsi;
	  for(vsi=mode_names.begin();vsi != mode_names.end();++vsi)
	  {
		cout << *vsi << endl;
	  }
	  cout << "Ifeat[0].rows,cols=" << Ifeat[0].rows << ","<<Ifeat[0].cols<<" learn_thresh:"<<learn_thresh<<endl;
  );
  vector<Mat>::iterator Iit;
  vector<string>::const_iterator mit;
  int num_models = 0;
  for (Iit = Ifeat.begin(), mit = mode_names.begin(); Iit != Ifeat.end(); ++Iit, ++mit)
  {
	  cout << "obj::learn_a_template"<<endl;
		Mat_<uchar>::iterator c = (*Iit).begin<uchar>();
		for(int i = 0;c != (*Iit).end<uchar>(); ++c,++i)
		{
			if(*c != 0&&*c != 1&&*c != 2&&*c != 4&&*c!=8&&*c!=16&&*c!=32&&*c!=64&&*c!=128)
				cout << i<<": bad value of c:"<<(int)*c<<endl;
		}

    OBJS_DEBUG_4(cout << *mit << ":" << endl;);
    if (modes.count(*mit) > 0) //We have models already for this mode
    {
      OBJS_DEBUG_4(cout << "Have models for this mode" << endl;);
      modes[*mit].learn_a_template(*Iit, Mask, session_ID, object_ID, framenum, learn_thresh, Score);
    }
    else //We have no models for this mode yet. Better insert one
    {
      OBJS_DEBUG_4(cout << "Learning a new model for this mode" << endl;);
      mmod_mode m(*mit);
      modes.insert(pair<string, mmod_mode> (*mit, m));
      OBJS_DEBUG_4(
    	  cout << "modes[*mit].mode = " << modes[*mit].mode << endl;
          cout << "  ... learn a template with the mode. learn_thresh: " << learn_thresh << endl;
      );
      modes[*mit].learn_a_template(*Iit, Mask, session_ID, object_ID, framenum, learn_thresh, Score);
    }
    num_models += (int) (modes[*mit].objs[object_ID].features.size());
    OBJS_DEBUG_4(cout << "num_models = " << num_models << endl;);
  }
  OBJS_DEBUG_2(cout << "# of templates for this object = "<< num_models << endl;);
  return num_models;
}

////////////////////////////////////////////////////////////////////////////////
// FILTERS
////////////////////////////////////////////////////////////////////////////////
/**
 * \brief For a given object, if the view index is not updated, update it so that
 * \brief framenum will return it's learned index
 * @param objname  Name of object for which we want an updated ViewIndex
 * @return number of total views for the object. -1 => error, no such object
 */
int mmod_filters::update_viewindex(string objname) {
    if (ObjViews.count(objname) <= 0) //We do not have this object in
    {
    	OBJS_DEBUG_2(cout << "update_viewindex() has no object = " << objname << endl;);
    	return -1;
    }
    //ObjViews do have this object
    int num_views = ObjViews[objname].size();
    if(num_views == 0)
    {
    	OBJS_DEBUG_2(cout << "update_viewindex() has ObjView[" << objname <<"].size() = 0" << endl;);
    	return -1;
    }
	vector<int>::iterator fnit = ObjViews[objname].frame_number.begin();  //frame number iterator
	vector<int>::iterator end_fnit = ObjViews[objname].frame_number.end();
	if(ViewIndex.count(objname) > 0)//ViewIndex already has this object name
	{
		int num_indices = (int)(ViewIndex[objname].size());
		if(num_views == num_indices)
		{
			OBJS_DEBUG_2(cout << "Have already indexed this object views file with " << num_views << " views."<< endl;);
			return num_views;
		}
		//We have a stale ViewIndex for this object, erase it
		ViewIndex.erase(objname);
		//Then refill it with framenum, index pairs
		for(int i = 0;fnit != end_fnit; ++fnit,++i)
		{
			ViewIndex[objname].insert(pair<int,int>(*fnit,i));
		}
    	OBJS_DEBUG_2(cout << "Done with update_viewindex, inserted "<<num_views<<" views" << endl;);
    }
    else //ViewIndex does not have an entry for objname
    {
    	multimap<int,int> mm;
    	for(int i = 0;fnit != end_fnit; ++fnit,++i)
    	{
    		mm.insert(pair<int,int>(*fnit,i));
    	}
    	ViewIndex.insert(pair<string, multimap<int,int> >(objname, mm));
    }
    return num_views;
}

/**
 * \brief For all objects, if the view index is not updated, update it so that framenum will return it's learned index
 * @return Total number of views in ObjViews
 */
int mmod_filters::update_viewindex()
{
	  map<string,mmod_features>::iterator it;
	  int num_views = 0;
	  for ( it=ObjViews.begin() ; it != ObjViews.end(); ++it)
		  num_views += update_viewindex((*it).first);
	  return num_views;
}

/**
 * \brief Match one modality in image I against learned object name and view (framenum) at R
 *
 * Called mainly from mmod_filters in mmod_objects.h
 *
 * @param I			Single modality input image, CV_8UC1
 * @param objname	Name of object to verify
 * @param R			Hypothesis of object location
 * @param framenum  View (or frame number)
 * @return			Matching score
 */
float mmod_filters::match_here(const cv::Mat &I, std::string objname, cv::Rect &R, int framenum)
{
	if(update_viewindex(objname) < 0)
		return 0.0;
	multimap<int,int>::iterator it;
	pair<multimap<int,int>::iterator,multimap<int,int>::iterator > eqrg; //equal range
	eqrg = ViewIndex[objname].equal_range(framenum);
	float score = 0, maxscore = -1;
	for(it = eqrg.first; it!=eqrg.second; ++it) //Go through multiple templates with same view (usually will only be one)
	{
		score = util.match_one_feature(I, R, ObjViews[objname], (*it).second);
		if(maxscore < score) maxscore = score;
	}
	return maxscore;
}

/**
 * \brief Learn a filter template: a std::map of objects and their features for their set of views. One modality only
 * @param Ifeatures		8UC1 binarized feature image for this mode (color or gradient, or ...). One bit set per pixel
 * @param Mask			8UC3 or 8UC1 silhoette of the object
 * @param objname		Name of this object class
 * @param framenum		Framenum (correlated to view)
 * @return				Number of views learned for this object. -1 => error
 */
int mmod_filters::learn_a_template( Mat &Ifeatures,  Mat &Mask, string objname, int framenum)
{
	cv::Mat temp;
	if (!Mask.empty()) //We have a mask, make sure it's 8UC1
	{
		if (Ifeatures.size() != Mask.size())
		{
			cerr << "ERROR: Mask in mmod_filters::learn_a_template size != Ifeatures image size" << endl;
			return -1;
		}
		if (Mask.type() == CV_8UC3)
		{
			//don't write into the Mask, as its supposed to be const.
			cv::cvtColor(Mask, temp, CV_RGB2GRAY);
		}
		else
			temp = Mask;
	}
	else
		return -1; //We require a mask
	//We're good. Fill up the features class:
	mmod_features f;
	cout << "In mmod_filters::Learn a template" << endl;
	int index = util.learn_a_template(Ifeatures, temp, framenum, f);
	if(ObjViews.count(objname)>0) //We already have this object, add to it
	{
		ObjViews[objname].insert(f, index);
	}
	else
	{
		ObjViews.insert(pair<string, mmod_features> (objname, f));
	}
	return (int)(ObjViews[objname].size());
}
//float match_one_feature(const cv::Mat &I, const cv::Rect &R, mmod_features &f, int index);

/**
 * \brief  After you've run Objs.match_all_objects(), You can use this to further filter recognitions according to the
 * \brief  learned filter model here.
 *
 * @param filt_features		This is the 8UC1 binarized feature image corresponding to this filter's modality
 * @param Objs				The learned object model which has just performed recognition using Objs.match_all_objects()
 *                          Objs's recognitions stored in rv, scores, ids, framed_nums, feature_indices will be altered
 *                          by this function's filtering.
 * @param thresh			The matching threshold for the filter
 * @return					Number of remaining matches
 */
int mmod_filters::filter_object_recognitions(const Mat &filt_features, mmod_objects &Objs, float thresh)
{
	int reclen = (int)Objs.rv.size();
	vector<string>::iterator nit = Objs.ids.begin();
	vector<Rect>::iterator rit = Objs.rv.begin();
	vector<float>::iterator scit = Objs.scores.begin();
	vector<int>::iterator fit = Objs.frame_nums.begin();
	vector<vector<int> >::iterator featit = Objs.feature_indices.begin();
	vector<float> fscores;
	for(int ij = 0; ij< reclen; ++ij)
	{
		float fscore = match_here(filt_features, Objs.ids[ij], Objs.rv[ij], Objs.frame_nums[ij]);
		fscores.push_back(fscore);
		if(fscore < thresh) //Too low
		{
			Objs.rv.erase(rit + ij);
			Objs.scores.erase(scit + ij);
			Objs.ids.erase(nit + ij);
			Objs.frame_nums.erase(fit + ij);
			Objs.feature_indices.erase(featit + ij);
			ij -= 1;
			reclen -= 1;
		}
	}
	if(!reclen)
	{
		cout << "All filter scores were squashed, they were:" << endl;
		vector<float>::iterator fsit = fscores.begin();
		for(;fsit != fscores.end();++fsit)
			cout << *fsit << ", ";
		cout << endl;
	}
	return reclen;
}
