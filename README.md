# ResNet

Residual Neural Networks ([ResNets](https://en.wikipedia.org/wiki/Residual_neural_network)) are very popular due to the residual layers that help fight the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).  The residual connection helps reduce the amount of gradients that are small or zero when training.

When used with convolutional layers, ResNets are very good at image classification.  In this project, I implemented a ResNet where you can input the number of residual layers, the size of the first input channel, and an expansion term.  I then created a model with and trained it on the [CIFAR10](https://en.wikipedia.org/wiki/CIFAR-10) dataset over 50 epochs.  The specific statistics I achieved for each class is shown in the following plot:

![ResNetAccuracy](https://user-images.githubusercontent.com/67863882/164306286-a779bae3-4b01-457a-a7df-baf8e71a8aa6.png)

The model was trained using an NVIDIA Tesla T4 GPU on Google Cloud Platform.

In addition, I plotted the [TSNE Visualization](https://en.wikipedia.org/wiki/Latent_space) in order to get a visualization of the data after being passed through the model. I plotted this at 1, 10, 25, and 50 epochs.  As the training progresses, you can see a separation in the representation vectors. Here is the visualization: 

![TSNE](https://user-images.githubusercontent.com/67863882/164309157-c25b3358-1489-4aac-8b9b-3cecf2e2ac1c.png)
