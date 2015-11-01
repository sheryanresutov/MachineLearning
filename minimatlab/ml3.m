%%% ECE 414 Project 3
%%% Sheryan Resutov, Eugene Sokolov, Harrison Zhao

clear all;close all;clc;

data = 'data_ml3.mat';
d = load(data);
theData = cell(4,2);

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


