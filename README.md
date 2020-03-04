# Metropolis-Hastings GAN experiments

Hi, I am using this repository to hold research/experimentation on Metropolis-Hastings
Generative Adversarial Networks (https://arxiv.org/pdf/1811.11357.pdf).

So far, I have been able to recreate results from the gaussian mixture experiment. Here is a particular example:

<img src="images/bad-nomh.png" width="400" height="400"> <img src="images/bad-mh.png" width="400" height="400">

For this example, I intentionally stopped the gan training before it was high quality, even as high quality as the 30 epoch
example in the original paper, so that I could test the feasibility of the MH approach in more likely scenarios, where the gan is not 
even close to optimal.
