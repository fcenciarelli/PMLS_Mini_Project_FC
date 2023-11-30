

# On-device Learning on ultra-low power microcontroller with CNN accelerator (MAX78000)

The project involves the creation of an on-device learning system that runs on the MAX78000 board. This board provides a cortex-M4 with a CNN engine supporting 1-, 2-, 4-, and 8-bit weights, with SRAM weight storage of 442KB and 512KB of data memory.
The on-device learning will be implemented by carrying out backpropagation on the layers of the neural network running on the device. 

## Aim

Obtain a system where starting with not well-trained or well tuned NN for KWS20 or MNIST on MAX78000, the board can be fed data and labels that will increase the accuracy of the model. (e.g. Train well with 9 keywords and then learn a 10th on the device itself by giving it data and carrying out training on-board.


## Steps

The project will involve the following steps:

*DONE* 1- **Training a neural network with KWS20 and MNIST to run on MAX78000**, using Maxim Integrated Pytorch framework to convert to code executable on the MAX78000 (Difficulty 2/10)

*STARTED* 2- **Carry out backpropagation** and weight update on-device on the last fully-connected (Difficulty 6/10)

 --- 2.1 Write functions in python to test 
 
 --- 2.2 Write the functions in C++ to execute them on the board

(Repeat for KWS20 and MNIST)


3- **Find a way to carry out backpropagation also on the convolutional layers**, not only the fully connected  (In theory it is possible to offload the weights by freezing the layer and memcpy -ing the weights) (Will probably run into memory issues with a lot of layers?) (Difficulty 9/10) (Maybe need to be using TinyEngine??)

--- 3.1 Write functions in python to test, hard to find kernels location maybe, but interesting challenge

--- 3.2 Write the functions in C++ to execute them on the board
(Repeat for KWS20 and MNIST)

(Repeat for KWS20 and MNIST)


4- Carry out **contribution analysis** on each layer for both sound and image data (to see how each layer training affects the model accuracy) (Difficulty 4/10)

5- Use contribution analysis to carry out sparse layer updates to remain in the memory constrains (Difficulty 8/10)


(6- Analyse how the accuracy varies with 1-, 2-, 4-, 8- bit weights if anything before does not work as planned) (Difficulty 4/10)



## Notes from drop-in Session 


- Result Section should tell the story of the development itself
- The kind of format should be: Document/Lesson Learned 
- Start writing very early-on, write code with while writing the draft itself






