# Project-BigData-Siamese-Model

#### Kaggle competition: Shopee - Price Match Guarantee *Full code / kaggle notebook is publicly available on Kaggle at www.kaggle.com/mvenou/productmatching-siamese-network)*
 ---

*In this Kaggle challenge our goal is to develop a discriminator network capabale of identifying unique products by partitioning a set of images into distinct groupings. During inference we are provided a dataset consisting of 70,000+ product images, along with perceptual hash codes and user-submitted descriptions. Some products may be featured in up to 50 images in the set and we are to return a CSV file listing each "post ID" and the ID from all images containing that same product.  Our training set contains 32,000+ of sample data.*

*This is a heavily imbalanced problem. On the data side, we have a classification problem with vastly more classes than positive examples per class. On the technological side we have unconstrained machinery during training, but face CPU / GRU budget constraints (8 hours CPU or 2 hours GPU) for inference. With a 70,000+ image inference data set, this resource limitation is significant.*

---
**Model Architecture**

My strategy is to build a One-Shot-Learning model similar to [1], comprised of a (resource-intensive) encoder and a very lightweight discriminator network.  This model is trained on "Siamese pairs" of examples, where each training sample consists of an anchor image and two comparison images- one image from same classification category and another image with a different classification. This Siamese network alleviates the enormous class imbalance within the dataset. However, instead of using a distance metric to produce a decision boundary as in [1], [2] and [3], I utilize a modified binary classification (sigmoid output) with loss function based on false positives and false negatives, which is discussed as an alternative option in [2]. 

Encodings for all of the test images will be produced and stored using a modestly sized network. We can then use an extremly light discriminator to categorize images into groups. In this way, while categorizing images may require a huge number of comparisons, this resource-intensive encoding process only occurs once per sample. On the other hand, classification of 70,000 images requires a great many pairwise comparisons that cannot benefit from GPU acceleration. Do to this, I limit my encoding model parameters to a fraction of that used in [3] and run inference on CPU. Encoder training is run on GPU because our resources are only limited during inference.

The encoder uses a pretrained Tensorflow Hub multi-lingual text embedding network [4] [5] [6] and a newly-trained CNN based on [1] to extract features from the image and its text description. The discriminator takes encoded images, encoded text, and perceptual hash of two images as input, and computes the "distance" along each of these features, and outputs an overall distance. 

*This model architecture and training strategy is based on "Learning a Similarity Metric Discriminatively, with Application to Face Verification" by Sumit Chopra Raia Hadsell Yann LeCun (2005) [1]. The choice of loss metric is based on DeepLearning.ai's Deep Learning Specialization course on Coursera [2], which credits [2]. See the bottom of this readme for additional citations*

---
**Data Pipeline**

During inference we will process (batches) of products through the encoder, one at a time. During training, however, products will be processed three at a time using the "Siamese" training structure. These "product triples" consist of an anchor product, a matching product and a non-matching product. The main effort in our data preprocessing is creating an efficient pipeline for the product triples of the form [(image1, title1, phash1). (image2, title2, phash2), (image3, title3, phash3)]

**Training**

Product triples are fed through the network in product pairs (anchor, match), (anchor, non-match) to yield "matching distance" and "non-matching distance." Our goal is to define a decision boundary using this distance metric. There are two approaches suggesed in [2]. One is to think in terms of a true distance metric (as in [1], [2] and [3]. The other approach is to treat this as a binary classification problem (match = 0 / non-match = 1) using a sigmoid output we can interpret as the probability that two images do not match. This idea is presented as an alternative approach in [2].

I chose to use binary classification (to let the network learn its own decision boundary) while maintaining the triplet / siamese structure (to deal with the large class imbalance). In this context our triplet loss function is "loss = (matching_dist + (1 - non_matching_dist))/2."

---
**My Journey**

This being my first Kaggle challenge, my first time working with a large unprepared dataset, and my first time facing serious computational GPU restrains, I learned a great deal working through the project. 

My first major obstacle was developing an efficient data pipeline, as my initial one produced data much too slowly to handle 70,000+ inference samples within a time-restricted environment. This difficulty arose from working with matched Siamese triples of data, which use of Tensorflow's high-level image processing pipeline tools. Instead, I learned to use 'tf.data.experimental.CsvDataset' and low-level data loading tools. This increased my data pipeline speed by a factor of 3 compared to my original implemenatation.

My second obstacle was in stripping down my encoder network to handle the processing and saving of 70,000+ images in a time-limited environment. My first models included elements of an object localization process and a pretrained Tensorflow Hub image feature extractor, both of which I sadly had to remove due to how much processing time they required. Instead I ended with en encoder model with under 40,000 paramaters, which is tiny compared to what I had been using in my Coursera deep learning specializations.

The third obstacle was cutting down the number of operations required to classify 70,0000+ products, when each category contained no more that 50 products. My initial, naive approach would require an impossible number of pairwise comparisons. This was very interesting and was my first encounter with using algorithm optimization (ideas I learned working when completing Foobar with Google) in a data science context.

The fourth obstacle was unexpected NaN results removing all image data from the model. Adding gradient clipping, batch norms and carefully checking all divisions did not remove the problem, which I still have been unable to track down. My makeshift "running out of time" solution is to babysit the training process and stop training before NaN's appear.

---
**Results**

Finally resolving the above challenges and leaving my model to train overnight, I awoke to find that the notebook had shut shut down, erasing my model checkpoints. I do not know why this occured, as I stayed within Kaggle's documented resource limits. After frantically retraining the model up until the last possible minute I found that, although the rules explicitly allowed the use of pretrained models, they prohibited me from installing Tensorflow Text that the Tensorflow Hub embedding [4] required in order to run. Dissapointed and without time to implement an alternative solution, I was not able to submit a solution in time for the competition deadline.

This of course exposes my own fault in not alloting myself an adequate amount of time to work through unanticipated issues that arose.  However, I learned a great deal throughout the process and have come out the better for it regardless of the ultimate outcome. 

One of my main takewaways is the importance of producing a minimal working end-to-end process built before spending too much time experimenting with architecture. I spent several days attempting to build an advanced model with novel ideas (nearly ALL of which I had to strip out after truly understanding my computational resource limitations). Had I first developed a minimal working model under the exact competition restrictions, I would have overcome the unexpected "nuts and bolts" obstacles with enough time remaining to produce a good model and satisfactory submission to the challenge.

---

**Citations:**
* [1] @INPROCEEDINGS{1467314,  author={Chopra, S. and Hadsell, R. and LeCun, Y.},  booktitle={2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)},   title={Learning a similarity metric discriminatively, with application to face verification},   year={2005},  volume={1},  number={},  pages={539-546 vol. 1},  doi={10.1109/CVPR.2005.202}}

* [2] @misc{author = {Andrew Ng}, title = {Special Applications: Face recognition & Neural Style Transfer}, howpublished = {Available at \url{https://www.coursera.org/learn/convolutional-neural-networks#syllabus} (2020/05/09)}}

* [3] @article{DBLP:journals/corr/SchroffKP15, author= {Florian Schroff and Dmitry Kalenichenko and James Philbin},  title = {FaceNet: {A} Unified Embedding for Face Recognition and Clustering},  journal = {CoRR}, volume    = {abs/1503.03832},  year={2015}}

* [4] Tensorflow Hub pretrained multilingial word embedding model, available at *https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3*, which in turn credits the two works cited below.

* [5] Yinfei Yang, Daniel Cer, Amin Ahmad, Mandy Guo, Jax Law, Noah Constant, Gustavo Hernandez Abrego , Steve Yuan, Chris Tar, Yun-hsuan Sung, Ray Kurzweil. Multilingual Universal Sentence Encoder for Semantic Retrieval. July 2019  

* [6] Muthuraman Chidambaram, Yinfei Yang, Daniel Cer, Steve Yuan, Yun-Hsuan Sung, Brian Strope, Ray Kurzweil. Learning Cross-Lingual Sentence Representations via a Multi-task Dual-Encoder Model. Repl4NLP@ACL, July 2019.
