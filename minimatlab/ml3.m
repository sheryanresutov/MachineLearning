%%% ECE 414 Project 3
%%% Sheryan Resutov, Eugene Sokolov, Harrison Zhao

clear all;close all;clc;

data = 'data_ml3.mat';
d = load(data);


%% - the data
plotDataML3(d.circles.x, d.circles.y)
title('the data - circles')
plotDataML3(d.bimodal.x, d.bimodal.y)
title('the data - bimodal')
plotDataML3(d.unimodal.x, d.unimodal.y)
title('the data - unimodal')
plotDataML3(d.spiral.x, d.spiral.y)
title('the data - spiral')

%% BIMODAL
x1 = 0:.01:20;
mu = [1 8];
s = 1;
phi = @(x, m) cos(2*pi*sqrt(x.^2));

x = d.bimodal.x;
y = d.bimodal.y;
logRegression(x,y,x1,mu,s,phi);
title('bimodal - logistic regression');             % bad classification

phi = @(x, m) cos(2*pi*0.2*sqrt(x.^2));
binGaussianGenerativeModel(x,y,x1,mu,s,phi);
title('bimodal - gaussian generative');            

%% UNIMODAL
x1 = -5:.01:5;
mu = [0, 0];
s = 0;
phi = @(x, m) x;

x = d.unimodal.x;
y = d.unimodal.y;
logRegression(x,y,x1,mu,s,phi);
title('unimodal - logistic regression')
binGaussianGenerativeModel(x,y,x1,mu,s,phi);
title('unimodal - gaussian generative');

%% SPIRAL
x1 = 0:.01:20;
mu = [1 9];
s = 4;
phi = @(x, m) exp(-(x-m).^2/(2*s^2)); % gaussian function
%phi = @(x, y) ; 

x = d.spiral.x;
y = d.spiral.y;
logRegression(x,y,x1,mu,s,phi);             % bad classification
title('spiral - logistic regression')
binGaussianGenerativeModel(x,y,x1,mu,s,phi);
title('spiral - gaussian generative');

%% CIRCLES
x1 = 0:.01:2;
mu = [1 1];
s = 1;
phi = @(x, m) exp(-(x-m).^2/(2*s^2)); % gaussian function

x = d.circles.x;
y = d.circles.y;
logRegression(x,y,x1,mu,s,phi);
title('circles - logistic regression')
binGaussianGenerativeModel(x,y,x1,mu,s,phi);
title('circles - gaussian generative');


