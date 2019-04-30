# attribute-label-embedding
An Implementation of Attribute Label Embedding (ALE) method for Zero-Shot Learning

!!! Run **$git lfs pull** to be able to pull data files with '.mat' extension.  
  
Validation phase is used for hyper-parameter tuning.   
Run **$python3 master.py --mode validation**

Test phase is used to train zero-shot learning model with fine-tuned hyper-parameters.   
Run **$python3 master.py --mode test**
   
Y. Xian, Christoph H. Lampert, B. Schiele, Z. Akata "Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly" TPAMI (2018). [[pdf]](https://arxiv.org/pdf/1707.00600.pdf) [[data link]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/)
