

> For such time-critical applications, real-time template matching is an attractive solution because new objects can be easily learned and matched online, in contrast to statistical-learning techniques that require many training samples and are often too computationally intensive for real-time performance [1], [2], [3],
[4], [5]. The reason for this inefﬁciency is that those
learning approaches aim at detecting unseen objects
from certain object classes instead of detecting a priori
known object instances from multiple viewpoints. The
latter is tried to be achieved in classical template
matching where generalization is not performed on
the object class but on the viewpoint sampling. While
this is considered as an easier task, it does not make
the problem trivial, as the data still exhibit signiﬁcant
changes in viewpoint, in illumination and in occlusion
between the training and the runtime sequence.

Individual objects rather than object classes are the focus of template matching. Hence, scaling doesn't matter as much? Well, there's a depth map which renders scaling-invariance less useful (true?).


> Each object is represented as a set of templates, relying on local dominant gradient orientations
to build a representation of the input images and
the templates.



## Questions from running the code

What does buildpyramid do? Seems to be related to mask so we only look at a small region in the image.