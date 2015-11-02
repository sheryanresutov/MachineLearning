%%% ECE 414 Project 3
%%% Sheryan Resutov, Eugene Sokolov, Harrison Zhao

clear all;close all;clc;

data = 'data_ml3.mat';
d = load(data);

%%
x1 = 0:.01:2;
mu = [1 1];
s = 1;
phi = @(x, m) exp(-(x-m).^2/(2*s^2)); % gaussian function

x = d.circles.x;
y = d.circles.y;
binGaussianGenerativeModel(x,y,x1,mu,s,phi);
title('circles');

%%
x1 = 0:.01:20;
mu = [1 8];
s = 1;
phi = @(x, m) x;

x = d.bimodal.x;
y = d.bimodal.y;
binGaussianGenerativeModel(x,y,x1,mu,s,phi);
title('bimodal');

%%
x1 = -5:.01:5;
mu = [0, 0];
s = 0;
phi = @(x, m) x;

x = d.unimodal.x;
y = d.unimodal.y;
binGaussianGenerativeModel(x,y,x1,mu,s,phi);
title('unimodal');

%%
x1 = 0:.01:20;
mu = [1 9];
s = 4;
phi = @(x, m) exp(-(x-m).^2/(2*s^2)); % gaussian function

x = d.spiral.x;
y = d.spiral.y;
binGaussianGenerativeModel(x,y,x1,mu,s,phi);
title('spiral');

%%
x1 = 0:.01:2;
mu = [1 1];
s = 1;
phi = @(x, m) exp(-(x-m).^2/(2*s^2)); % gaussian function

x = d.circles.x;
y = d.circles.y;
logRegression(x,y,x1,mu,s,phi);
title('circles');

%%
x = d.circles.x;
y = d.circles.y;
plotDataML3(d.circles.x, d.circles.y)
title('circles')
plotDataML3(d.bimodal.x, d.bimodal.y)
title('bimodal')
plotDataML3(d.unimodal.x, d.unimodal.y)
title('unimodal')
plotDataML3(d.spiral.x, d.spiral.y)
title('spiral')


