# dlcourse.ai

### Open Deep Learning Course: dlcourse.ai (from ODS.ai)

![Course Banner](https://dlcourse.ai/assets/images/img_banner.jpg)

The course is designed to deal with modern deep learning from scratch and does not require knowledge of neural networks or machine learning in general. Lectures by streams on Youtube, tasks in Python, discussions and help in the best Russian-speaking ML-communities - [ODS.ai](http://ods.ai/) and [ClosedCircles](http://closedcircles.com/?invite=a5f6bea89716a16054cfbfb3fafa6ed111dff4b8).

At the same time and in the same volume, the course is read for master students of **Novosibirsk State University**, as well as students of the CS center of Novosibirsk.

## Lectures and Assignments
 
| Lesson | Description           | Links  |
|-------------------|-------------|:-----:|
|**Lecture 1: Introduction** | What is the course about, what is machine learning and deep learning, the main domains are computer vision, NLP, speech recognition, reinforcement learning. Resources. | [Video](https://www.youtube.com/watch?v=_q46x0tq2FQ)<br>[Slides](https://www.dropbox.com/s/veif179mw6cdp2v/Lecture%201%20-%20Intro.pptx?dl=0) |
|**Seminar 1: Python, numpy, notebooks** | A brief overview of the toolkit required for the course - Python, Jupyter, numpy. Google Colab as a Jupyter Notebook runtime in the cloud. | [Video](https://www.youtube.com/watch?v=9lPrQEAQSqA) [Notebook](https://colab.research.google.com/drive/1FBdo0TAv5eiWNl909vrcAQeau476rlOK) |
|**Lecture 2: Elements of machine learning** | Review of the supervised learning problem. K-nearest neighbor as an example of a simple learning algorithm. Training and test samples. Hyperparameters, their selection using validation set and cross-validation. General sequence of actions for training and validation of models (Machine Learning Flow). | [Video](https://www.youtube.com/watch?v=1BUuB28FDOc)<br>[Slides](https://www.dropbox.com/s/h1r9iju8i1c1gyp/Lecture%202%20-%20Machine%20Learning%20-%20annotated.pptx?dl=0) |
|**Seminar 2: Setting up the environment** | Setting up the environment necessary for solving assignments. Some details of KNN.| [Video](https://www.youtube.com/watch?v=jnjvku8-zkc)|
|**Assignment 1, Part 1: K-nearest neighbors** | Getting familiar with Python and numpy, implementing the K-nearest neighbor classifier by hand. Hyperparameter selection using cross-validation. | [Assignment](assignment1)|
|**Lecture 3: Neural Networks** | Linear classifier is a single layer neural network. Softmax, cross-entropy loss function. Stochastic gradient descent training, weight regularization. Multilayer neural networks, fully-connected layers. Backpropagation algorithm. | [Video](https://www.youtube.com/watch?v=kWTC1NvL894)<br>[Slides](https://www.dropbox.com/s/ywn9xoxeyy7250b/Lecture%203%20-%20Neural%20Networks%20-%20annotated.pdf?dl=0) |
|**Seminar 3: Gradient calculation** | A detailed review of the calculation of softmax and cross-entropy gradients.| [Video](https://www.youtube.com/watch?v=bZihskzsSjM)|
|**Assignment 1, Part 2: Linear classifier** | Do-it-yourself linear classifier implementation, gradient counting and training with SGD.| [Assignment](assignment1)|
|**Lecture 4: PyTorch** | Backpropagation with matrices. An introduction to PyTorch. Weighing initialization. Improved gradient descent algorithms (Adam, RMSProp, etc.). | [Video](https://www.youtube.com/watch?v=tnrbx7V9RbA)<br>[Slides](https://www.dropbox.com/s/bdk2rdjxx4c0cte/Lecture%204%20-%20Framework%20%26%20Details%20-%20annotated.pdf?dl=0) |
|**Assignment 2, Part 1: Neural Networks** | Implementing your own multilayer neural network and training it.| [Assignment](assignment2)|
|**Lecture 5: Neural Networks in practice** | GPUs. Training process and overfitting / underfitting in practice. Learning rate annealing. Batch Normalization. Ensembles.| [Video](https://www.youtube.com/watch?v=2gIn9cVn9cA)<br>[Slides](https://www.dropbox.com/s/fa047fxlbqcmv96/Lecture%205%20-%20Neural%20Network%20In%20Practice%20-%20annotated.pdf?dl=0)|
|**Assignment 2, Part 2: PyTorch** | Implementation of a neural network in PyTorch, practice of training and visualizing model predictions.| [Assignment](assignment2)|
|**Lecture 6: Convolutional Neural Networks** | Convolution and pooling layers. Evolution of architectures: LeNet, AlexNet, VGG, ResNet. Transfer learning. Augmentation.| [Video](https://www.youtube.com/watch?v=tOgBz8lFz8Q)<br>[Slides](https://www.dropbox.com/s/k8rtpvlc3xaj65b/Lecture%206%20-%20CNNs%20-%20annotated.pdf?dl=0)|
|**Assignment 3: Convolutional Neural Networks** | Implementation of Convolutional Neural Networks by hand in PyTorch.| [Assignment](assignment3)|
|**Lecture 7: Segmentation и Object Detection** | More complex computer vision tasks are segmentation and object detection.| [Video](https://www.youtube.com/watch?v=r2KA99ThEH4)<br>[Slides](https://slides.com/vladimiriglovikov/title-texttitle-text-17#/)|
|**Assignment 4: Hotdog or Not** | Using transfer learning and fine tuning methods on the example of hotdog recognition.| [Assignment](assignment4)|
|**Lecture 8: Metric Learning, Autoencoders, GANs** | Metric Learning on the example of face recognition, an overview of some of the methods of unsupervised learning in DL.| [Video](https://www.youtube.com/watch?v=ajEQ10s8XRg)<br>[Slides](https://www.dropbox.com/s/n25eai8ivlq60bh/Lecture%208%20-%20Metric%20and%20Unsupervised.pdf?dl=0)|
|**Lecture 9: Introduction to NLP, word2vec** | A brief overview of the field of natural language processing and the application of deep learning to it using word2vec as an example.| [Video](https://youtu.be/MBQdMQUZMQM)<br>[Slides](https://www.dropbox.com/s/na7lpz9xhgx8gp1/Lecture%209%20-%20Intro%20to%20NLP%20-%20annotated.pdf?dl=0)|
|**Assignment 5: Word2Vec** | PyTorch implementation of word2vec on a small dataset.| [Assignment](assignment5)|
|**Lecture 10: Recurrent Neural Networks** | Application of recurrent neural networks in natural language recognition problems. LSTM architecture details.| [Video](https://www.youtube.com/watch?v=tlj-CMibdMI)<br>[Slides](https://www.dropbox.com/s/eafd6z6sr2ajnka/Lecture%2010%20-%20RNNs%20-%20annotated.pdf?dl=0)|
|**Assignment 6: RNNs** | Using LSTM for Part of Speech Tagging| [Assignment](assignment6)|
|**Lecture 11: Audio and speech recognition** | Application of deep learning methods to speech recognition problem. A quick overview of other audio tasks.| [Video](https://www.youtube.com/watch?v=JpS0LzEWr-4)<br>[Slides](https://www.dropbox.com/s/tv3cv0ihq2l0u9f/Lecture%2011%20-%20Audio%20and%20Speech.pdf?dl=0)|
|**Lecture 12: Attention** | Using the Attention mechanism in NLP on the example of a machine translation task. Transformer architecture, modern development.| [Video](https://www.youtube.com/watch?v=qKL9hWQQQic)<br>[Slides](https://www.dropbox.com/s/1nk66rixz4ets03/Lecture%2012%20-%20Attention%20-%20annotated.pdf?dl=0)|
|**Lecture 13: Reinforcement Learning** | An introduction to reinforcement learning, using deep learning techniques. Basic Algorithms - Policy Gradients and Q-Learning| [Video](https://www.youtube.com/watch?v=_x0ASf9jV9U)<br>[Slides](https://www.dropbox.com/s/txh5ujn4een98t0/Lecture%2013%20-%20Reinforcement%20Learning%20-%20annotated.pdf?dl=0)|
|**Assignment 7: Policy Gradients** | Solution of the RL - Cartpole model problem using the REINFORCE algorithm based on Policy Gradients.| [Assignment](assignment7)|
|**Lecture 14: More on Reinforcement Learning** | Model-based RL using AlphaZero as an example. Criticism and some possible ways of development of the region.| [Video](https://www.youtube.com/watch?v=aOIK1i1xt_M)<br>[Slides](https://www.dropbox.com/s/gv6pc7v26jw8e8i/Lecture%2014%20-%20More%20RL%20-%20annotated.pdf?dl=0)|
|**Lecture 15: Conclusion** | Results. Things to do after the course to increase the amount of Deep Learning in your life.| [Video](https://www.youtube.com/watch?v=V9TuLKhaDqQ)<br>[Slides](https://www.dropbox.com/s/t14f0eiyxednlpa/Lecture%2015%20-%20Outro%20-%20annotated.pdf?dl=0)|
