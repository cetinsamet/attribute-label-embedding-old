# attribute-label-embedding
An Implementation of Attribute Label Embedding (ALE) method for Zero-Shot Learning.   
  
There are 32 classes; 20 training classes + 12 test classes.    

*On validation phase*, only 20 training classes are used and they are split as 15 training classes + 5 test classes. Hyper-parameter tuning is performed only on this phase.   
Run **$python3 master.py --mode validation**    
    
*On test phase*, all 32 classes are used and they are split as 20 training classes + 12 test classes.    
Run **$python3 master.py --mode test**  
  
You can run *prepareData.ipynb* script to preprocess data located in APY/ directory and create ready-to-use data that will be placed in APYP/ directory. Or you can follow the [[link]](https://drive.google.com/drive/folders/1ZEEYDnxdCk30h7KkxdXNV339UpPkLjNC?usp=sharing) to download data.
  
     
Y. Xian, Christoph H. Lampert, B. Schiele, Z. Akata "Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly" TPAMI (2018). [[pdf]](https://arxiv.org/pdf/1707.00600.pdf) [[data link]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/)
