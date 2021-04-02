# Reproduction project on "Learning Where to Focus for Efficient Video Object Detection" by Jiang et al.

Written by. Dani Tóth and Ruben van Oosterhoudt

# Introduction 

Just a short introduction about the project and the paper.

- Motivation and scope
- No generic first sentence
- 3 contributions
- figure 1 (visual abstract of the paper)
- Few research

# Original research paper

Object detection becomes a more prominent aspect in our day to day live. With face recognition 
for unlocking phones, ball tracking during a football match and autonomous driving. All of these 
applications have one thing in common, motion. Motion decreases performance of object detection 
because of occlusion, rare poses, and motion blur. To compensate this we need the temporal 
information of the object. In earlier work optical flow is used to determine the temporal 
information, but this has its limitations. Optical flow is time-consuming, with only 10 frames 
per second (FPS). With the example of autonomous driving this could cause problems. 

So, the writer of the paper introduced, Learnable Spatio-Temporal Sampling (LSTS) to tackle this 
problem. With this approach high-level features will be propagated across frames to predict the 
location. By sampling specific location from the feature maps F_t and F_t+k, which are extracted 
from I_t and I_t+k. Where I_t is the current frame, and I_t+k is the next frame. Then, similarities 
between feature maps will be used to determine weights needed for propagating between F_t and 
F_t+k to produce an F’_t+k. This could be iterated to propagate across multiple frames. The High- 
and Low-feature maps are enhanced by two proposed methods: Sparsely Recursive Feature Updating 
(SRFU) and Dense Feature Aggregation (DFA), respectively. These methods will be discussed later. 

# Dataset used

ImageNet is a well known name within the machine learning community. With 14 million images, 
at least 1 million bounding boxes and 20 thousand categories. The large amount of 
data results in a wide variety of applications. For the paper they use a similar dataset, 
ImageNet VID. The images are replaced by 3862 training videos, 30 object 
categories and boundary boxes. Examples are shown in figure ???. Because of storage restricting 
during reproducibility project a training set of 1910 training videos is used.

ImageVID/ImageNET
- What is the dataset made off?
    - Classes
    - Annotations
- Where is it originated from
- Benefits of the dataset


# Machine learning model used.

Residual Networks, or ResNet for short, was introduced by researchers at MicroSoft Research 
in 2015. This architecture tackled the problem with vanishing/exploding gradient caused by 
large number of layers in convolutional neural networks (CNN) architectures. In ResNet, they 
implement a so-called skip connection which can skip a certain number of layers within the 
CNN. Implementing this skip connection it is possible to skip sections that have a negative 
influence on the performance. With this technique deeper CNNs could be created. In the paper 
the ResNet101 is used, ResNet is the architecture and 101 are the number of layers. ResNet 
is implemented to extract the feature maps from the images a stated in the introduction. For 
the High-Level features the complete ResNet will be used, while for Low-Level features only 
a part will be used. This cut-off point is notated as “Conv4_3” and can be determined with 
figure ???.  

ResNet101
- goal of ResNET
- pretrained model
- till layer 50 for low level features
    - implement image for layer selection
- complete for high level features

# Own implementation

Show what we implemented on our own.
- Software environment that we used (Python, pytorch)

# Results

Show the results of our own implementation 


# Conclusion

Here comes the conclusion of the paper

# Discussion

In the introduction of the paper two terms are given: keyframe and non-keyframe. 
With the naming we can figure out that the keyframe should be more important. 
But the question still stands, why is that specific frame more important. 
And after looking in the code became clear that a keyframe is every tenth frame, and the 
rest are non-keyframes.

Here comes the discussion of the paper.
-   Flaws of the original paper.
    - Not all steps are clearly stated in the paper.
    
-   Flaws of the github.
    - Outdated function are used to do certain processes.



# The machine learning reproducibility checklist

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
