---
layout:     post
title:      逻辑回归(Logistic Regression)
subtitle:   关于逻辑回归的简单梳理
date:       2020-10-19
author:     Dorzii
header-img: img/post-bg-swift2.jpg
catalog: true
tags:
    - 机器学习
    - 分类问题
---



(啊这，第一篇博客就从逻辑回归开始吧)

逻辑回归是机器学习中较为基础的线性分类算法，多用于二分类问题，主要就是线性平面套了一层$Sigmoid$函数的壳，使得输出概率为[0，1]，从而表示出样本处于正类的概率。

## 定义

对于线性可分的样本空间$D$，$D=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\},x_i \in R^n,y_i\in\{0,1\}$，决策边界为$w^T+b=0$，假设某个样本点$h_w(x)=w^T+b>0$则认为其属于正类,下图以二分类为例:

然而此时$h_w(x)=w^T+b$是连续的，最终的预测结果是离散变量，考虑到概率也是一种离散值，可以将$h_w(x)$往概率上转换，再设定概率的阈值从而使连续值变为离散值。
逻辑回归的做法是给$h_w(x)$套上一层$Sigmoid$函数:
$$P(Y=1|x)=sigmoid(h_w(x))= \frac{1}{1+e^{-(w^Tx+b）}}$$
若$h_w(x)\rightarrow+\infty，P(Y=1|x)\rightarrow1，h_w(x)\rightarrow-\infty$，$P(Y=1|x)\rightarrow0$，此时可估计样本所属类别的概率。

## 损失函数

确定模型函数后，此时的目的是求解模型中的参数。在统计学中，常常使用极大似然估计法来求解，即找到一组参数，使得在这组参数下，我们的数据的似然度（概率）最大。并且这里对最大似然函数取对数后，是关于$(w,b)$的高阶连续可导凸函数，可以方便通过一些凸优化算法求解，比如梯度下降法、牛顿法等，所以此时损失函数就与最大似然函数有关。

设：
$$P(Y=1|x)=p(x)，P(Y=0|x)=1-p(x)$$
则似然函数为：
$$L(w)=\prod\limits_{i=1}^{N}[p(x_i)]^{y_i}[1-p(x_i)]^{1-y_i}$$
取对数：
$$\begin{aligned}
lnL(w)&=\sum\limits_{i=1}^{N}{y_iln{p(x_i)}+{(1-y_i)ln{(1-p(x_i))}}}\\
&=\sum\limits_{i=1}^{N}{y_iln\frac{p(x_i)}{1-p(x_i)}+ln(1-p(x_i))}\\
&=\sum\limits_{i=1}^{N}{y_i(w^Tx_i+b)-ln(1+e^{w^Tx_i+b})}\end{aligned}$$
事实上为了方便可以将权值向量与输入向量进行扩充，$w=(w^{(0)},w^{(1)},\dots,w^{(n)},b)^T\\$，$w=(x^{(0)},x^{(1)},\dots,x^{(n)},1)$则上式简化为：
$$lnL(w)=\sum\limits_{i=1}^{N}{y_i(w^Tx_i)-ln(1+e^{w^Tx_i})}$$
对整个数据集取平均似然损失(添加负号是为了变为最下化问题，可以进行梯度下降等算法)可以得到：
$$J(w)=-\frac{1}{N}\sum\limits_{i=1}^{N}{y_i(w^Tx_i)-ln(1+e^{w^Tx_i})}$$

## 反向传播推导

TODO

## 代码

TODO

