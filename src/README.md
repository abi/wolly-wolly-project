

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
```C
  if(..)
  {
```

# Resources

Real time, scalable object recognition

Mentor: Gary Braski, Willow Garage (bradski@willowgarage.com)

Project background:

This project will be to put together two existing types of object recognition systems together in a way that will complement both to achieve fast, and scalable object recognition. Both techniques are based on simple, easy to implement ideas below.
Specific tasks:

The first technique is "Linemod" *Multimodal Templates for Real-Time Detection of Texture-less Objects in Heavily Cluttered Scenes (Oral) * /IEEE International Conference on Computer Vision (ICCV), Barcelona, Spain, November 2011./ http://campar.cs.tum.edu/pub/hinterstoisser2011linemod/hinterstoisser2011linemod.pdf Which binarizes features into 8 bits over a grid using gradient images and depth images. We have code that can run many 10's of thousands of these vectors over an image per second.

The second technique is "winner take all": “The Power of Comparative Reasoning”, Jay Yagnik , Dennis Strelow, David Ross , Ruei-Sung Lin, /International Conference on Computer Vision/ (2011). research.google.com/pubs/archive/37298.pdf This is a "recipe" for creating meta-features by permuting other feature vectors M times and taking the max of the first K terms of the resulting vectors.

The problem with 1 is that it slows down linearly with each new object or object view learned. Also, these features are sparse, so they don't fit into any existing scaling algorithm (such as Locality Sensitive Hashing LSH, or approximate nearest neighbors). Using 2, we can solve this problem: Randomly create N Linemod feature vectors. Match them against a patch (perhaps scaled by using a depth image). From these random match results, create M new vectors by (a) randomly selecting K non-repeating linemod responses, and selecting the max of those K's as that feature. (b) Do a M times for the full feature vector. This creates a dense response map that can be directly fed into the FLANN library for scalable learning. The result will, I think, be very fast, scalable and publishable.
Criterion:

Basic knowledge about image processing
Be familiar with OpenCV


## Code documentation

### mmod_color

HLS Features and Gradient features

Depth features are only used for the objects in data/imgs.



# TODO

* Fix bug mentioned in email
* Train the LINE-MOD algorithm on the test data