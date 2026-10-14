# Rebuilding-DL - Chapter 3 : First "Image" Generator (GAN)






## Goal 

Normally, a GAN uses convolutionnal networks (We will see that in the next chapters...), and train itself with / generates pictures (1920x1080 values between 0 and 255. 3 times for each red, green, blue proportion). If I try to train this on my computer, it explodes.

I just want to understand how it works. So, I'll simplify the goal. Instead of generating 3x1920x1080 numbers to look like manga drawings, I will train only 2 coordinates to look like a circle. 



![dataplot](README_figures/20000%20epochs%20-%20500%20batchsize%20-%200,01%20lr.png) 





## Logic

Let's consider 2 neural networks : 
- ```D```, the discriminator, gets 2 coordinates and must predict if the given point belongs to the circle. It is a classical classification network.
- ```G```, the generator, takes 2 noises and gives 2 coordinates.

The goal is to train ```G``` to trick ```D``` by making it predict its generated coordinates as belonging to the circle.

The two will train in parallel. ```G``` must learn to trick ```D``` and ```D``` must learn to not be tricked by ```G```





## How to implement it ?

### Training loop
- First, we train ```D```.
  - We compose a data batch, with few points from the true circle, labelled ```1```, and we generate a bunch of points with ```G```, labelled ```0```. 
  - we predict some label with ```D```, and apply the classical back propagation.
- Second, we train ```G```. 
  - We generate false data with ```G```, but wse label them ```1```. 
  - When we give them to ```D```, it can predict : 
    - ```1``` : ```D``` has been tricked. ```G``` had sucess in tricking it. 
    - ```0``` : ```D``` has not been tricked. the loss will be big here, and the back propagation on ```G``` will affect highly the coefficients.

### Visualisation

In this code, we implemented visualisation of the progress of the training of ```G```. For some key epoch numbers, we plot a bunch of points it generated to compare the alignment with the true circle. This is the picture you can see at the top of this readme file. 