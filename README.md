# conditionalGAN_TF
The repository provides the training codes for Conditional Generative Adversarial Networks on MNIST dataset using TensorFlow.  

The idea of CGANs is to impose a condition on both generator and discriminator with some extra information from labels (y).

The implementation is based on the original paper [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf).  

CGAN addresses the problem with GANs and DCGANs, i.e. zero control over the type of images that are generated. The control is regained by conditioning both the generator and discriminator on the class label y.

The advantages of CGAN include:

- The generator produces realistic samples for each label by parameterizing the learning process.  
- The discriminator learns to discriminate between fake and real samples by leveraging the label information.  
- Using the conditional auxiliary information, the generator and discriminator continue to generate and classify images, respectively. 

The results from CGAN on MNIST are shown below:


![res1](https://user-images.githubusercontent.com/26203136/184329701-48c84905-b6b0-4af5-8bc4-3844d6b2ce19.png)

The code uses the following packages  
Tensorflow  
Numpy  
Matplotlib  
Graphviz  
Sklearn  
