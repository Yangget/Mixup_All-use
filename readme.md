# MixUP

This is a simple and easy to use mixup with examples and explanations.

## Explanation
The mixup neighborhood distribution can be understood as a data enhancement approach that makes the model appear linear when dealing between regions between samples and samples.We believe that this linear modeling reduces the incompatibility of predicting data outside of the training sample.From the principle of Occam's razor, linearity is a good inductive bias because it is one of the simplest possible behaviors. Figure 1 shows that the mixup causes a linear transition of decision boundaries from one class to another, providing a smoother estimate of uncertainty. Figure 2 shows the average performance of two neural network models trained on the CIFAR-10 dataset using the two methods of mixup and ERM. Both models have the same structure and are evaluated using the same training process on the same sample randomly sampled from the training data. Models trained with mixup are more stable when predicting data between training data.  
More detail please read https://arxiv.org/abs/1710.09412
![fig1](https://github.com/Yangget/Mixup_All-use/blob/master/images/Fig1.png)
![fig2](https://github.com/Yangget/Mixup_All-use/blob/master/images/Fig2.png)
## Algorithm

```python
l = np.random.beta(alpha, alpha, size)

X_l = l.reshape(size, 1, 1, 1)
y_l = l.reshape(size, 1)

X = X1 * X_l + X2 * (1 - X_l)
Y = Y1 * y_l + Y2 * (1 - y_l)
```

## How to use

```python
from Mixup import mixup

batch_x, batch_y = mixup(alpha, batch_x, batch_y)
```

## Performance

+ alpha = 0.5
![0.5](https://github.com/Yangget/Mixup_All-use/blob/master/result/batch_x_alpha(0.5).jpg)  
+ alpha = 0.6
![0.6](https://github.com/Yangget/Mixup_All-use/blob/master/result/batch_x_alpha(0.6).jpg)  
+ alpha = 0.7
![0.7](https://github.com/Yangget/Mixup_All-use/blob/master/result/batch_x_alpha(0.7).jpg)  
+ alpha = 0.8
![0.8](https://github.com/Yangget/Mixup_All-use/blob/master/result/batch_x_alpha(0.8).jpg)  
+ alpha = 0.9
![0.9](https://github.com/Yangget/Mixup_All-use/blob/master/result/batch_x_alpha(0.9).jpg)  
