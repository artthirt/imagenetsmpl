# imagenetsmpl
Sample code for train NN for ImageNet<br>

Pretrained model can be loaded by https://drive.google.com/file/d/0B9-nwoybwuRJUGtweHlFZm1DTTQ/view?usp=sharing
Model pretrained only for 220 classes<br>
<pre>
Model:
  convolution:
   1 ->  size of weight - 7x7, stride - 4, channels input 3, kernels - 64, without maxpooling, uses relu
   2 ->  size of weight - 5x5, stride - 1, channels input 64, kernels - 256, with maxpooling, uses relu
   3 ->  size of weight - 3x3, stride - 1, channels input 256, kernels - 512, with maxpooling, uses relu
   4 ->  size of weight - 3x3, stride - 1, channels input 512, kernels - 1024, without maxpooling, uses relu
   mlp:
   1 -> input features - 4096 (2x2x1024) , output features - 4096, uses relu
   2 -> input features - 4096, output features - 2048, uses relu
   3 -> input features - 2048, output classes - 1000, uses softmax

Used:
imagenetsmpl -gpu -pass number -batch number -load2 <path/to/model(model.bin_ext)> [-f /path/to/imagenet/dir -lr <learing rate>]\
              [-image /path/to/image(for predict classes)] [-images /path/to/dir/with/images(for predict all images in directories)]

</pre>
