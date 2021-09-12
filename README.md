[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- ABOUT THE PROJECT -->
## About The Project
In order to study generated magnetic fields qualitatively and quantitatively various techniques can be
employed. One of them is proton radiography (deflectometry). It allows to measure both electric and magnetic fields and even track their evolution (on a ps scale).
The protons, initially flying from a point-like source with some small divergence, pass through field and are deflected. They form some kind of a pattern on a detector, and studying this pattern may provide impormation on field's value and distributions. 
This is a difficult multi-parameter reverse task. To solve it, the simulations are carried out. They include:
1) Assumption on the current carrying circuit. Without some initial idea of this contour, the number of solutions to the problem is infinite.
2) Electromagnetic field calculation.

![image](https://user-images.githubusercontent.com/90211042/133005202-787bbd6d-022b-4c02-adb1-161fdc595b20.png)

3) [Proton flight simulation](https://github.com/kofeinix/Ballistic_deflectometry) with field value iteration.
4) Synthetic images comparation with experimental one for all field values. Finding the best match.
5) Current carrying circuit modification. Go to (1) and repeat until the difference between images is the smallest.

This script deals with step 4. Manual comparation of the images upon big synthetic image sample number takes extremely long time. Furthermore, the criterea for comparation is quite limited - e.g. width and hight of some characteristic area, which neglects other characteristical features that may be present. See examles for manual comparation below:
![image](https://user-images.githubusercontent.com/90211042/133004443-0568ba44-3105-425d-8a8f-45164bcdc478.png)

The more advanced approach, implemented here, is training a neural network on synthetic images, using the desired field values as labels. Then using the predict() on the experimental image will provide the field value, that could produce such an image. In this script, a Convolutional Neural Network (CNN) is implemented and trained.

## Input images
Input data are 500x500 (resised in the code to 256x256) binary images. They are created for a single current carrying circle, but the current and potential are varied in a wide range. 
In can be noted, that both produced at [step 3 ](https://github.com/kofeinix/Ballistic_deflectometry) images and of course experimental images are not binary. However if they are compared right away, without preliminal pre-processing, the parameters extraction will have much bigger error. Thus all images are binarized by some script, not presented here.
The images are located at .\2_parameters_data folder, and one experimental imaged is added to demonstrate the similarity between synthetic and experimental data.
525 images are present for training, 143 for validation and 42 for final testing. 

## CNN structure
The CNN architecture is used, as it allows to detect similar patterns in different part of the data arrays, thus it is resistant to shift in structures in images. 
* The CNN consists of three convolutional layers, each followed by a pooling layer. 
* ReLu is used as activation function in the CNN
* CNN output is flattened and foies to 10-neurons dense layer, and then to the 2 output parameters.
* For final layers, Sigmoid activation is used.
* Loss function is mean squared error and Optimizer is Adam.

## Output data
In the output folder, the trained model is saved, if so chosen in the beginning of the code.
The loss over the epoch images are also saved, along with the file stating the RMS error in absolute values (of current and electric potential).
Checkpoints and training log are saved in the .\output\checkpoint folder.

## Training results
The results and a trained model are presented in a .\output\saved folder. It can be seen, that the current RMS is several kA, and the potential is several kV, with is acceptable taking in account the range of the current (40-160 kA) and potential (0-200 kV). 
![proton_deflect_demo_with_shifts](https://user-images.githubusercontent.com/90211042/133005129-0019b675-24eb-4a33-bff9-9487b59d2a15.png)

## Notes
_The work is still in progress and some improvements and corrections may be soon done._
_The article is being prepared for submission._

## Contact

Iurii Kochetkov -  iu.kochetkov@gmail.com

Project Link: [https://github.com/kofeinix/Ballistic_deflectometry](https://github.com/kofeinix/Ballistic_deflectometry)

<!-- MARKDOWN LINKS & IMAGES -->

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/iu-kochetkov/
