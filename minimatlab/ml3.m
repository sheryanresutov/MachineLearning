%%% ECE 414 Project 3
%%% Sheryan Resutov, Eugene Sokolov, Harrison Zhao

clear all;close all;clc;

data = 'data_ml3.mat';
d = load(data);
i = 0;
figure
hold on;
for i = 1:size(d.circles.x(:,1))
	if(d.circles.y(i,1) == 0)
		plot(d.circles.x(i,1),d.circles.x(i,2), 'r.')
	elseif(d.circles.y(i,1) == 1)
		plot(d.circles.x(i,1),d.circles.x(i,2), 'b*')
	end
end
hold off;

figure
plot(d.bimodal.x(:,1),d.bimodal.x(:,2), 'b*')
figure
plot(d.unimodal.x(:,1),d.unimodal.x(:,2), 'b*')
figure
plot(d.spiral.x(:,1),d.spiral.x(:,2), 'b*')

