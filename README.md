# Reproduction project on "Learning Where to Focus for Efficient Video Object Detection" by Jiang et al.

Written by Dani Tóth and Ruben van Oosterhoudt

![Figure 1](https://github.com/DaniTot/DL_reproduction/blob/main/Images/Network.jpg)

## Introduction 
For the course Deep Learning at the Technical University of Delft we were given the task to replicate a 
paper published in the field. This to create a better understanding in Deep Learning, gaining experience in reproducing a paper and comparing personal experience to the paper of Edward Raff [[1](https://arxiv.org/abs/1909.06674)]. For the project we chose “Learning Where to Focus for Efficient 
Video Object Detection” by Jiang et al [[2](https://arxiv.org/abs/1911.05253)].

## Original research paper

Object detection becomes a more prominent aspect in our day to day life. With examples like face recognition for unlocking phones, ball tracking during a football match and autonomous driving. All of these applications have one thing in common, motion. Motion decreases performance of object detection, in comparison to images, because of occlusion, rare poses, and motion blur. To compensate for this we need the temporal information of the object. In earlier work optical flow is used to determine the temporal information, but this has its limitations. Optical flow is time-consuming, with only 10 frames per second (FPS). With the example of autonomous driving this could cause problems.

To tackle this problem Dense Feature Aggregation, DFA for short, is introduced to determine objects in videos [[3](https://arxiv.org/abs/1703.10025)]. The writers proposed a technique where feature maps are aggregated with the help of flow fields. They take a random image I<sub>i</sub> and its neighbour I<sub>j</sub>. With these images they can create a flow field using a flow network. This indicates the flow between the images. From I<sub>j</sub> they create a feature map F<sub>j</sub>. A bilinear warping function is applied on the flow field and a feature map to create feature maps warped from frame j to frame i. Dense aggregation is adding all the warped feature maps with a calculated weight factor to create an aggregated feature map. This could then be used to detect objects within the frame with for example blur.

Sparse Recursive Feature Updating, SRFU for short, is an improvement to DFA [[4](https://arxiv.org/abs/1711.11577)]. Because DFA uses all the images of the video it is relatively slow. The paper proposes to only take the keyframes into account. Keyframes are recursive with an interval of 10. The information from the current keyframe features aggregates with the old keyframe, which holds all information of the previous keyframe, and propagates to the next keyframe.

In the paper “Deep Feature Flow for Video Recognition” from Zhu et al [[5](https://arxiv.org/abs/1611.07715v2)]. they proposed the principles of keyframes. Here they state that a keyframe can be seen as a starting frame. With a non-keyframe being, in case of this paper, the next frame in the video. When a feature map is created of both frames the translation of the object of interest could be determined. The paper states that the keyframe should be changed in regular intervals to increase accuracy and speed. For the ImageNet VID dataset they set this interval on 10, the same as what is stated in the code.

So, the writer of the paper introduced Learnable Spatio-Temporal Sampling (LSTS) to tackle this problem. LSTS will replace relatively long calculation times of the flow fields with predicting the next frame with learned weights within DFA and SRFU. The structure is shown in figure 1. The features F<sub>t</sub> and F<sub>t+k</sub> will be extracted from current image, I<sub>t</sub>, and the next image, I<sub>t+k</sub>. A point is taken from F<sub>t+k</sub> and will be compared to the surrounding points in F<sub>t</sub>, as shown in figure 2. By similarity comparison, between the point and the surrounding, the affinity weights could be calculated. This weight displays the flow between the current and the previous weight. With the weights and F<sub>t</sub> the predicted F’<sub>t+k</sub> could be calculated. This can then be propagated through training and returns an improved surrounding needed to predict F’<sub>t+k</sub>. If the network is trained it could increase FPS and accuracy.

The architecture is implemented in Python 2, with the MXNet library.


## ImageNet VID

ImageNet is a well known name within the machine learning community. With 14 million images, at least 1 million bounding boxes and 20 thousand categories. The large amount of data results in a wide variety of applications. For the paper they use a similar dataset, ImageNet VID. The images are replaced by 3862 training videos, 30 object categories and boundary boxes. Because of storage restrictions during the reproducibility project a training set of 1910 training videos is used. 

To demonstrate the problem of video object recognition, we took some frames of the dataset. In figure 2a and 2b we see that the objects are obstructed by being partially out of the frame and being obstructed by several trees. Both figure 2c and 2d are showing motion blur caused by moving of the camera or object of interest respectively. These images are examples of problematic images during object recognition.

![Figure 2](https://github.com/DaniTot/DL_reproduction/blob/main/Images/ExampleImages.jpg)

## ResNet101

Residual Networks, or ResNet for short, was introduced by researchers at MicroSoft Research in 2015. This architecture tackled the problem with vanishing/exploding gradient caused by a large number of layers in convolutional neural networks (CNN) architectures. In ResNet, they implement a so-called skip connection which can skip a certain number of layers within the CNN. Implementing this skip connection it is possible to skip sections that have a negative influence on the performance. With this technique deeper CNNs could be created. In the paper the ResNet101 is used, ResNet is the architecture and 101 is the number of layers. ResNet is implemented to extract the feature maps from the images as stated in the introduction. For the High-Level features the complete ResNet will be used, while for Low-Level features only a part will be used. This cut-off point is notated as “Conv4_3” and can be determined with the figure shown below. During the reproduction a pre-trained ResNet101 from TorchVision is used for the feature extraction.

![Figure 3](https://github.com/DaniTot/DL_reproduction/blob/main/Images/ResNET.jpg)

## Own implementation

For the framework of our reproduction, we chose Python 3 with the PyTorch library. 

Given a batch size, a number of videos from the ImageNet training set are selected randomly. Due to a limitation of our implementation, only videos with a resolution of 1280x720 can be processed. The frames are loaded one-by-one, and resized to 1000x562. Each frame is normalized, and classified as a key or non-keyframe, with a fixed keyframe interval of 10, so every 10th frame is considered a keyframe, while all the rest are considered non-keyframes. We extract high-level features from the keyframes (via ResNet101), and low-level features from the non-keyframe (via ResNet101, up until Conv4_3). The feature space from keyframes then have a shape of (1024, 18, 32), and the feature space from non-keyframes have a shape of (512, 71, 125).

#### LSTS
Learnable Spatio-Temporal Sampling (LSTS) is the heart of this architecture, and it works as follows:

It takes two feature maps, F<sub>0</sub> and F<sub>1</sub>. N number of fractional sampling locations p<sub>n</sub> are randomly generated around each location of the feature map (where n = 1, 2, …, N). The number of channels of both feature maps is reduced from 1024 to 256 via the embedding function f, which is a single convolutional layer, with single stride, no padding, and a kernel size of 1. Then, for each location p<sub>0</sub> on the feature space F<sub>1</sub>, we repeat the following procedure:

###### Similarity weights 

The similarity s(p<sub>n</sub>) between features p<sub>n</sub> of F<sub>0</sub> and feature p<sub>0</sub> on F<sub>1</sub> is computed via dot product. Note that since p<sub>n</sub> is fractional, the corresponding values on F<sub>0</sub> must be interpolated. Bilinear interpolation was used:

<img src="https://latex.codecogs.com/svg.image?f(F_0)_{p_n}=&space;\sum_q&space;G(p_n,q)\cdot&space;f(F_0)_q">

where q are the locations on the feature map f(F<sub>0</sub>), and 

<img src="https://github.com/DaniTot/DL_reproduction/blob/main/Images/G_eq.png" width="475">

Large positive products mean high similarity, and large negative products mean low similarity. We get the normalized weights S(p<sub>n</sub>), by normalizing s(p<sub>n</sub>) by the sum of its absolutes for each sampling location pn. The prediction for feature of F<sub>1</sub> at p<sub>0</sub> is calculated by aggregating the sampled features F<sub>1</sub>(p<sub>n</sub>) with the corresponding similarity weights:

<img src="https://latex.codecogs.com/svg.image?F'_1(p_0)=&space;\sum_N&space;s(p_n)\cdot&space;f(F_0)_{p_n}">

###### Backprop

Gradient of (non-normalized) similarity weights at p_0: 

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;s(p_n)}{\partial&space;p_n}=&space;\sum_q&space;\frac{\partial&space;G(p_n,q)}{\partial&space;p_n}\cdot&space;f(F_0)_q\cdot&space;f(F_1)_{p_0}">

Note that F<sub>0</sub> and F<sub>1</sub> (and consequently f(F<sub>0</sub>) and f(F<sub>1</sub>)) are 2D feature spaces, so p<sub>n</sub> will have an x component, and a y component. Thus the derivative of the bilinear kernel will be:

<img src="https://github.com/DaniTot/DL_reproduction/blob/main/Images/delta_G_x_eq.png" width="550">

<img src="https://github.com/DaniTot/DL_reproduction/blob/main/Images/delta_G_y_eq.png" width="550">

In the updating phase, the sampling locations p_n are then offset based on the gradient and a scalar to magnify the offset:
<img src="https://latex.codecogs.com/svg.image?p'_n=p_n&plus;s(p_n)&space;\text{\hspace{1cm}&space;for&space;}n=1,&space;2,&space;...,&space;N">

###### Aggregation

Once the above is calculated for each p<sub>0</sub>, we get the prediction F'<sub>1</sub>. F<sub>1</sub> and F'<sub>1</sub> are passed through a series of 3 convolutional layers (with shapes of  (3x3x256), (1x1x16), (1x1x1) respectively) and then a softmax, in order to produce the corresponding weights. The LSTS output is then

<img src="https://latex.codecogs.com/svg.image?F_{out}=&space;W_1\circ&space;F_1&plus;W_{pred}\circ&space;F'_1">


#### SRFU and DFA
In the Sparsely Recursive Feature Updating (SRFU) a memory feature space F<sub>mem</sub> is created and updated with the LSTS output F<sub>out</sub> from each keyframe and the previous keyframe. This is done at every keyframe interval (10), when LSTS is ran with the new keyframe, and the previous keyframe from 10 frames before (i.e. F<sub>0</sub> is the feature space from the old keyframe, and F<sub>1</sub> is the feature space from the new keyframe). The memory feature space F<sub>mem</sub> is then updated with the output F<sub>out</sub>.

In Dense Feature Aggregation the memory feature space is propageted to predict the next non-keyframe. This is achived by running the LSTS module with the most recent F<sub>mem</sub> and the low level features extracted from the non-keyframe. Note that the shape of the low level feature space will be different that of the high level one. Thus, it is first transformed with a series of three convolutional layers (with shapes (3x3x256), (3x3x512), (3x3x1024) respectively, and with ReLU operators in between), such that its shape would resemble that of the high-level feature space. And so, in LSTS, F<sub>0</sub> will be the memory feature space, and F<sub>1</sub> will be the transformed feature space from the non-keyframe.



At this point, the program kept crashing, likely due to some kind of memory error in the gradient calculation. We were unable to find and fix the bug, and so we could not complete the reproduction. 


## Results

+ The weight generation and aggregation in LSTS is very. It is hypotised that this is due to our failure to vectorize many of the operations, which remained nested for loops.
+ Calculating the gradients in LSTS causes a memory error, that we were unable to fix by the time of submitting this paper


## Conclusion

Here comes the conclusion of the paper

## Discussion

We started this paper by introducing the reason behind this blogpost, looking how much we could reproduce given the information of the paper. During the reproduction we came across some problems that are noteworthy: missing explanations, outdated code and discrepancies between code and the paper.

An example of missing a proper explanation are the concepts of keyframe and non-keyframe. With the naming we could figure out that keyframe is superior to a non-keyframe, but what makes this difference? With the code it became clear that a keyframe happens once every 10 frames. Which only raised more questions, why 10? After reading several other papers it became clear that the keyframes and non-keyframe were used for SRFU and DFA to reduce the number of aggregations and so reduce computation. The number 10 is found as an optimal for ImageNet by a empirical study.

Our first step of reproducing was installing and running the code linked to the paper. With the given Windows and Linux installer we couldn't figure out how to make this work. After this setback and communication with the other groups with the same paper we made the decision to make a personal interpretation of the code. During the replication process we found out that a lot of the functions used were outdated or custom made by the author. Without much prior knowledge in coding, in what we thought was C, we couldn't replicate these functions.

During the reproduction with the code we came across a few discrepancies between the original paper, and their implementation. Two convolution networks that we replicated, Low2High and Quality network, had discrepancies in their layers. For example, the Low2High network code states that the should exist out of 3 layers: (1x1x256), (3x3x256) and (3x3x1024). While the paper says that it should be (3x3x256), (3x3x512), (3x3x1024). Also functions that were stated in paper and code deviated from each other. Example is the similarity weight normalization we found in the code. Where they use the softmax function, while the paper says 

<img src="https://latex.codecogs.com/svg.image?S(p_n)=&space;\frac{s(p_n)}{\sum_N&space;s(p_n)}"> 

These relatively small differences made it difficult to check whether we were doing the right thing. 

A paper from Edward Raff showed the most relevant factors for a reproducible paper, for us that is readability, code availability and hyperparameters specified. Just as Raff describes, readability of the paper was the biggest obstacle during the reproduction process. It took us several reads before the basics of the paper were clear. And while code included isn’t a big factor, for reproducibility of the paper, it could give valuable information that wasn’t clear from the paper. Combining these factors to find the hyperparameters and reasoning behind them were quite unclear from the original paper.

## References

1] [“A Step Toward Quantifying Independently Reproducible Machine Learning Research” by E. Raff.](https://arxiv.org/abs/1909.06674)

2] ["Learning Where to Focus for Efficient Video Object Detection" by Jiang et al.](https://arxiv.org/abs/1911.05253)

3] [“Flow-Guided Feature Aggregation for Video Object Detection” by Zhu et al.](https://arxiv.org/abs/1703.10025)

4] [“Towards High Performance Video Object Detection” from Zhu et al.](https://arxiv.org/abs/1711.11577)

5] [“Deep Feature Flow for Video Recognition” from Zhu et al](https://arxiv.org/abs/1611.07715v2)


## The machine learning reproducibility checklist

for all models and algorithms presented, check that you include: 
- A clear description of the mathematical setting, algorithm, and/or model.
- An analysis of the complexity (time, space, sample size) of any algorithm.
- A link to a downloadable source code with specification of all dependencies, including external libraries.

for any theoretical claim, check that you include:
- A statement of the result
- A clear explanation of each assumption
- A complete proof of the claim

for all figures and tables that present empirical results, check that you include:
- A complete description of the data collection process, including sample size.
- A link to a downloadable version of the data set or simulation environment.
- An explanation of any data that was excluded and a description of any preprocessing step.
- An explanation of how samples were allocated for training, validation, and testing.
- The range of hyperparameters considered, method to select the best hyperparameter configuration, and specification of each hyperparameter used to generate results.
- The exact number of evaluations runs.
- A description of how experiments were run.
- A clear definition of the specific measure or statistics used to report results.
- Clearly defined error bars. 
- A description of results with central tendency (e.g., mean) and variation (e.g., stddev).
- A description of the computing infrastructure used.

## Information of Discussion 

Here comes the discussion of the paper.
+ Flaws of the original paper.
    + Many essentials steps/details are missing from the paper (i.e. the definition of the embedding function used).
    + We suspect that LSTS is meant to be a type of transformer model, applied to the sequence of embeded featuers of video frames. In case it is indeed the case, this was not emphasized in the paper.
+ Flaws of the github.
    + Dependency list is missing.
    + Instructions and automated installer only work for Linux.
    + Windows installer is broken.
+ Discrepancies in the implementation details between the original paper, and their implementation:
    + Low2high transformation convolution network: 
        + code says: (1x1x256), (3x3x256), (3x3x1024)
        + paper says: (3x3x256), (3x3x512), (3x3x1024)
    + Quality network:
        + code says: (3x3x256), (3x3x16), (3x3x1)
        + paper says: (3x3x256), (1x1x16), (1x1x1)
    + Similarity weight normalization:
        + Code says: softmax
        + Paper says: <img src="https://latex.codecogs.com/svg.image?S(p_n)=&space;\frac{s(p_n)}{\sum_N&space;s(p_n)}">
