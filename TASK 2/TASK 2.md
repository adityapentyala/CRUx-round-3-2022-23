**TASK 2 PROBLEM STATEMENT**

Gradient Descent + Genetic Algorithm

Dataset: CIFAR-10

1. Understand how gradient descent and backpropagation work (as long as you're comfortable with the chain rule of differentiation, you should be good). 
2. Understand and implement the TinyVGG architecture using any framework of your choice (Pytorch/Tensorflow).
3. Tune your hyperparameters using the same genetic algorithm used in round 2.
4. Brownie points: 
  a. Experiment with custom architectures, and compare and analyze the results.
  b. Add a pre-trained model to your experiment(s) to improve its performance.
  c. The task entails you tuning the hyperparameters of your layers, not the number of layers and the architecture. Tuning these as well will be treated as a bonus.
 
 ----------------------------------------------------------------------------------------------------------------
 _NOTE: Included PDF copy of console output of test run 2 to visualize best network architectures and hyperparameters used._
 
 Each generation of the algorithm consists of 12 individuals, of which 6 are selected as elites to populate the next
 generation. Final results of each test run compared with baseline model. All accuracies shown of test runs refer to
 testing accuracies.
 
 HYPERPARAMETERS TUNED:
 1. Number of convolutional layers: {1,2}
 
 		Hyperparameters tuned for each convolutional layer:
		
		a. Kernel size: {2, 3, 4, 5, 6}
		
		b. Number of filters: {32, 64, 128, 256}
		
		c. Strides: {1, 2, 3, 4, 5}
		
		d. Activation function: {RELU, Sigmoid}
		
 2. Number of maxpool layers: {1, 2}
 
 		Hyperparameters tuned for each maxpool layer:
		
		a. Pool sizes: {1, 2, 3}
		
 3. Number of dense layers: {1, 2}
 
 		Hyperparameters tuned for each Dense/fully connected layer:
		
		a. Number of nodes: {256, 512, 1024}
		
		b. Activation function: {RELU, Sigmoid}
		
 4. Number of epochs: {3, 4, 5, 6} (Test run 1); {3, 4, 5, 6, 7, 8} (Test run 2)

 5. Optimizer: {Adam, RMSProp, SGD}

BASELINE MODEL:

72.85% training accuracy, 63.60% testing accuracy

Number of epochs: 10

Baseline model architecture:

<img width="407" alt="image" src="https://user-images.githubusercontent.com/62715046/222783518-56169bc5-3cd6-49c2-941c-58fc26750473.png">

<img width="566" alt="image" src="https://user-images.githubusercontent.com/62715046/222783639-6a99cc67-c2bd-481a-a7f6-e932fb59e499.png">


TEST RUN 1: (Original + 8 generations)

Time Taken: over 4 hours, 30 minutes

<img width="944" alt="Screenshot 2023-03-02 163507" src="https://user-images.githubusercontent.com/62715046/222778369-1c0e2c9c-be84-4d7e-a5dc-613fbf29d662.png">

<img width="944" alt="Screenshot 2023-03-02 163545" src="https://user-images.githubusercontent.com/62715046/222778435-af776c7a-73fd-475c-a329-a5df04932b51.png">

<img width="946" alt="Screenshot 2023-03-02 163633" src="https://user-images.githubusercontent.com/62715046/222778480-9db06208-2333-4caa-b2cb-ac60e7220e82.png">

<img width="428" alt="Screenshot 2023-03-02 163821" src="https://user-images.githubusercontent.com/62715046/222778539-c8fa7db2-b014-487d-9a09-4fd06de64328.png">

![Figure_1](https://user-images.githubusercontent.com/62715046/222785248-c125b26b-2bd3-44d4-905a-d5f73f32fe71.png)


TEST RUN 2: (Original + 10 generations)

Time taken: 8 hours, 10 minutes

<img width="499" alt="Screenshot 2023-03-03 215746" src="https://user-images.githubusercontent.com/62715046/222781250-d6e59e96-2827-4838-bb3f-9da83bb1230d.png">

<img width="486" alt="Screenshot 2023-03-03 215835" src="https://user-images.githubusercontent.com/62715046/222781270-5b942ffb-0f8f-432e-b832-a2bacc621a48.png">

<img width="452" alt="Screenshot 2023-03-03 215923" src="https://user-images.githubusercontent.com/62715046/222781293-79eab5e6-825d-40d8-a5cf-2497cba8135c.png">

<img width="512" alt="Screenshot 2023-03-03 215957" src="https://user-images.githubusercontent.com/62715046/222781328-09d4dbf1-89b4-4a44-8074-e4145da51cab.png">

<img width="477" alt="Screenshot 2023-03-03 220054" src="https://user-images.githubusercontent.com/62715046/222781352-1be6ad9e-4b2a-45d1-8113-651577b6a3d2.png">

![Figure_1](https://user-images.githubusercontent.com/62715046/222781389-b7d9b807-0478-4acb-bcaa-993f16fbc6fd.png)
