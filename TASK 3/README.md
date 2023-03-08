**TASK 3 PROBLEM STATEMENT**

SEMI SUPERVISED LEARNING

Dataset: CIFAR-100 (with 33% randomly labeled data, rest unlabeled)

1. Understand and implement semi-supervised learning: self-learning.
2. Use any VGG architecture.
3. Use only the functional/modular API in TensorFlow (or) PyTorch. Create a callback (TF) that allows you to stop training 
   if your model isn't improving, or build it in your training loop in PyTorch (or use pytorch lightning if time allows you 
   to learn that).
4. Use keras-tuner or raytune for automating your hyperparameter tuning

-----------------------------------------------------------------------------------------------------------------------

Self-learning, or self-training, is a proxy label method of semi supervised learning in which a model is trained on a limited labelled training set. Then, the model is used to predict labels of an unlabelled training set. If the probability assigned to the output class is higher than a predetermined threshold, signifying the model's "confidence" in its prediction, the datapoint and its pseudo-label are added to the labelled training set. The model is then re-trained over the new training set, and this process continues until the model is no longer able to generate confident pseudo-labels.

In my code, I have implemented this training-pseudolabelling cycle in the following way:

First, the hyperparameters of the model are tuned by keras.tuner, in a bid to maximise early testing accuracy.

Then, a predetermined number of iterations for the training cycle are set. The model is trained on the training set for 5 epochs. It then assigns as many confident pseudolabels as it can on an unlabelled training set. If the number of new pseudolabeled datapoints is non-zero, the cycle is repeated unless the number of predetermined iterations of the cycle have been reached. At any point during the epoch training, if the validation accuracy drops, or the positive difference between successive validation accuracies is less than 0.1%, the epoch training is stopped and the cycle skips to the pseudolabelling step.

As an additional countermeasure against the common drawback of self learning, that is, the tendency of a self-learning model to amplify its own mistakes during pseudolabelling leading to a sizeable bias, the confidence threshold is incremented by a constant fraction as set by the user, upto a maximum of 0.85 or 85%.

-----------------------------------------------------------------------------------------------------------------------

TEST RUN 1: Iterations = 7
_NOTE: Excerpts of the output are given here. A complete copies of the console output are provided in the containing folder of this file._

<img width="220" alt="image" src="https://user-images.githubusercontent.com/62715046/223140202-edc68aab-1b88-4839-a5b2-9bbcf5315127.png">

<img width="404" alt="image" src="https://user-images.githubusercontent.com/62715046/223140274-bdd23ac0-295b-4610-97d5-520fee886a77.png">
<img width="407" alt="image" src="https://user-images.githubusercontent.com/62715046/223140385-6f3e81f1-b105-40e5-b6ba-98838fca1107.png">

_Tuned hyperparameters and architecture of the model._


![Figure_1](https://user-images.githubusercontent.com/62715046/223141086-b1494cdb-beda-485e-a1b5-8ba0d16e423c.png)

_Training and testing accuracies over epochs. In this figure, an "Epoch" refers to a single training and psuedolabelling cycle._


TEST RUN 2: Iterations = 15, Epochs (per training run) = 7

<img width="323" alt="image" src="https://user-images.githubusercontent.com/62715046/223629406-c50ac5f2-ec11-4c61-82ee-b550ca9edb61.png">

<img width="428" alt="image" src="https://user-images.githubusercontent.com/62715046/223629511-60f5e6df-50c8-4d8c-844d-4adb6bc9f206.png">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/62715046/223629551-b122b065-6f49-49b6-b4ec-3a9e3bb5dc27.png">

_Tuned hyperparameters and architecture of the model._


![Figure_2](https://user-images.githubusercontent.com/62715046/223629624-d9b85548-bc9f-453d-a96b-b3de61b6fd2d.png)

_Training and testing accuracies over epochs. In this figure, an "Epoch" refers to each individual (completed) epoch of the 15 training runs._


From the two graphs, we can see that while the training accuracy continued to improve steadily, the validation accuracy reached a plateau at about 20.5%.
