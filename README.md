# Reproduction project on "Learning Where to Focus for Efficient Video Object Detection" by Jiang et al.

Written by Dani Tóth and Ruben van Oosterhoudt

# Introduction 
For the course Deep Learning at the Technical University of Delft we were given the task to replicate a 
paper published in the field. This to create a better understanding in Deep Learning and gaining 
experience in reproducing a paper. For the project we chose “Learning Where to Focus for Efficient 
Video Object Detection” by Jiang et al.

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
location. By sampling specific location from the feature maps F<sub>t</sub> and F<sub>t+k</sub>, 
which are extracted from I<sub>t</sub>, and I<sub>t+k</sub>. Where I<sub>t</sub> is the current frame, 
and I<sub>t+k</sub> is the next frame. Then, similarities between feature maps will be used to determine 
weights needed for propagating between F<sub>t</sub> and F<sub>t+k</sub> to produce an F’<sub>t+k</sub>. 
This could be iterated to propagate across multiple frames. The High- and Low-feature maps are enhanced 
by two proposed methods: Sparsely Recursive Feature Updating (SRFU) and Dense Feature Aggregation (DFA), 
respectively. These methods will be discussed later. 

# ImageNet VID

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


# ResNet101

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
the figure shown below. During the reproduction a pre-trained ResNet101 from TorchVision is used for the 
feature extraction.

![Figure 3](https://github.com/DaniTot/DL_reproduction/blob/main/Images/architecture_of_ResNet.png)



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

During starting phase of our project we tried running the original GitHub code to have a better understanding 
of the structure. With the given Windows and Linux installer we couldn’t figure out how to make the code run. 
After this setback we started replication the complete code of the GitHub and tried to make a personal 
interpretation of the code. Soon after this we found out that a lot of function used are outdated or even 
non-existing, which made replicating significantly harder. 

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
