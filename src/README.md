

# Data

Download the data from http://vault.willowgarage.com/wgdata1/vol1/Bradski/CS231a_2011/ and put it in a data directory, which is not included for space reasons. And imgs.tar.gz from http://vault.willowgarage.com/wgdata1/vol1/Bradski/ which is much larger and contains 25 different objects

# Install

Install Homebrew if you don't already have it. The installation seems to be pretty fast possibly because some packages are cached by `brew` in a remote repo so you don't have to re-compile.

```sh
brew install opencv
brew install boost
make
make abi
./a.out (Should print Hello World!)
```

# Coding guide

Follow the existing code style in the existing codebase. This entails the following important things:

* 4 spaces soft tabs
* Comments that look like Java comments
* Use the debug macros that are already defined by Gary
* Brace for block statements on the next line with the same indentation level

  if(..)
  {

# TODO

* Fix bug mentioned in email
* Train the LINE-MOD algorithm on the test data
* Fix this compilation error:

```
g++ -arch i386 -g -I./ -DSTANDALONE_DEBUG -MM mmod_color.cpp > mmod_color.dep
g++ -arch i386 -g -I./ -DSTANDALONE_DEBUG -o mmod_color.o -c mmod_color.cpp 
g++ -arch i386 -g -I./ -DSTANDALONE_DEBUG -MM mmod_feature.cpp > mmod_feature.dep
g++ -arch i386 -g -I./ -DSTANDALONE_DEBUG -o mmod_feature.o -c mmod_feature.cpp 
g++ -arch i386 -g -I./ -DSTANDALONE_DEBUG -MM mmod_general.cpp > mmod_general.dep
g++ -arch i386 -g -I./ -DSTANDALONE_DEBUG -o mmod_general.o -c mmod_general.cpp 
mmod_general.cpp: In member function ‘int mmod_general::learn_a_template(cv::Mat&, cv::Mat&, int, mmod_features&)’:
mmod_general.cpp:972: error: invalid initialization of reference of type ‘const cv::Mat&’ from expression of type ‘std::vector<cv::Point_<int>, std::allocator<cv::Point_<int> > >’
/usr/local/include/opencv2/imgproc/imgproc.hpp:809: error: in passing argument 1 of ‘cv::Rect cv::boundingRect(const cv::Mat&)’
mmod_general.cpp: In member function ‘int mmod_general::score_with_ground_truth(const std::vector<cv::Rect_<int>, std::allocator<cv::Rect_<int> > >&, const std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >&, std::string&, cv::Mat&, int&, int&, int&, cv::Rect&)’:
mmod_general.cpp:1083: error: invalid initialization of reference of type ‘const cv::Mat&’ from expression of type ‘std::vector<cv::Point_<int>, std::allocator<cv::Point_<int> > >’
/usr/local/include/opencv2/imgproc/imgproc.hpp:809: error: in passing argument 1 of ‘cv::Rect cv::boundingRect(const cv::Mat&)’
make: *** [mmod_general.o] Error 1
```