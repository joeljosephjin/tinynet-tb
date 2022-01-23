# Tuberculosis diagnosis with a CNN

This repo contains the implementation of the convolutional neural network
for tuberculosis diagnosis described in
[Automated diagnosis of Tuberculosis from Chest X-Ray Screening using Deep Network Architecture and Visualization](),
which I will call tinynet for short. The network uses frontal chest X-Rays
images as input.

You can run the network running simply

```bash
python3 train.py
```

If you want to run a cross-validation study (5-fold), you can run:

```bash
python3 train.py --cross-validation
```
