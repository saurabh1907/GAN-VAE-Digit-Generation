<h1>Handwritten Digit Generation using VAE & GAN <h1>


<h2>Overview</h2>
In this literature, I intend to do a comparative analysis of performance of Generative Adversarial Network (GAN) vs Variational Autoencoder (VAE) to generate MNIST Handwritten Digits. Generative Models can learn any distribution of data through unsupervised learning and has shown great results in recent times. I will be discussing two main approaches Generative Adversarial Networks (GAN) and Variational Autoencoders (VAE). Generative Adversarial Network aims at attaining an equilibrium between Discriminator and Generator whereas Variational Autoencoder aims at maximization of lower bound of log-likelihood of data. I further tested VAE and different variations of GAN architecture including DCGAN and CDCGAN model on MNIST dataset and CelebA dataset.


<h2> How to run</h2>

- Run the python file corresponding to the model Eg. python VAE-MNIST.py for generating images with Variational Encoders-
- The dataset can be downloaded by enabling Download=true in corresponding models


<h3> Dataset</h3>
I am using two datasets- MNIST and CelebA. The MNIST dataset imply for the Modified National
Institute of Standards and Technology dataset. This dataset having 28×28 pixel grayscale image of
70,000 small square of digit between 0 and 9 which are handwritten single digits.
PyTorch will give us entry to access the MNIST data set easily. CelebA is another dataset I am using
to extent the applications of GAN. It is a collection of 200,000 + facial images of celebrities. It has
been downloaded from Kaggle and extracted to project directory. The image samples are transformed
which includes converting image to tensors, normalizing data around mean and variance

<h3> Defining the Learning Parameters </h3>

- Batch Size defines the no. of examples in forward/backward pass. As the MNIST images
are very small (28×28 greyscale images), using a larger batch size is not a problem. I am
using 128. Using very high batch size speeds up training but can run into Out of Memory
error.
- Epochs is no. of times entire dataset is passed through NN. I have used 200 epochs.
- Sample-size that defines fixed size noise vector that I will feed into our generator. Using the
noise vector, the generator will generate the fake images.
- Latent is the latent vector or the noise vector size. The input feature size for the generator is
going to be the same as this latent vector size.
- k is a hyperparameter that indicates the number of steps to put in to the segregator
- Computation device- CPU or a GPU
- Mean and Median of 0.5 for image normalization
- Learning rate of 0.0002 for gradient descent. Very small value can lead to slow convergence.
