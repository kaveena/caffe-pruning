# triNNity-caffe

Caffe is a deep learning framework made with expression, speed, and
modularity in mind. It is developed by Berkeley AI Research
([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning
Center (BVLC) and community contributors.

This is a customized distribution of Caffe developed by the Software
Tools Group at Trinity College Dublin. It is maintained as a fork that
cleanly merges over the BVLC Caffe master branch.

triNNity-caffe includes:

- Dynamic LR adjustment (triangular and triangular2)
- Saliency Computation during training
- Flexible Quantization to arbitrary bit precision
- Pruning (Sparsification), both pointwise and channel pruning
- General quality-of-life fixes and performance improvements

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
