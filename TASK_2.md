**TASK 2 PROBLEM STATEMENT**
Gradient descent + genetic algorithm
Dataset: CIFAR-10
1. Understand how gradient descent and backpropagation work (as long as you're comfortable with the chain rule of differentiation, you should be good). 
2. Understand and implement the TinyVGG architecture using any framework of your choice (Pytorch/Tensorflow).
3. Tune your hyperparameters using the same genetic algorithm used in round 2.
4. Brownie points: 
    a. Experiment with custom architectures, and compare and analyze the results.
    b. Add a pre-trained model to your experiment(s) to improve its performance.
    c. The task entails you tuning the hyperparameters of your layers, not the number of layers and the architecture. Tuning these as well will be treated as a bonus.
    
Tiny VGG architecture implemented using tensorflow. Architecture of network is as follows:
<img width="432" alt="image" src="https://user-images.githubusercontent.com/62715046/222136890-f04191cd-302d-4991-81aa-bd34790034cb.png">

"Tiny" refers to the scaled down size of a stereotypical VGG architecture. Where VGG architectures usually involve 10+ weight layers, the above architecture uses 4.

