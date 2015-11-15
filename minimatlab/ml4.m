%%% ECE 414 Project 3
%%% Sheryan Resutov, Eugene Sokolov, Harrison Zhao
clear all;close all;clc;

kvar = 0.2;
k = @(x1,x2) exp(-abs(x1-x2).^2/(2*kvar));

numxs = 1000;
sigmaN = 0.2;
x = linspace(0,1,numxs);
y = sin(2*pi*x);

subplotNumber = 1;
for numSamples = [1 2 4 25]
    xt = rand(1,numSamples);
    noise = normrnd(0,sigmaN,1,numSamples);
    t = sin(2*pi*xt) + noise;
    
    C = zeros(numSamples);
    for ii=1:numSamples
        for jj=1:numSamples
            C(ii,jj) = k(xt(ii),xt(jj)) + (ii==jj)*sigmaN;
        end
    end
    Cinv = inv(C);
    mu = zeros(1,numxs);
    var = zeros(1,numxs);
    for ii=1:numxs
        kx = k(x(ii),xt);
        mu(ii) = kx*Cinv*t';
        var(ii) = k(x(ii),x(ii))-kx*Cinv*kx';
    end
    subplot(2,2,subplotNumber);
    hold on;
    subplotNumber = subplotNumber + 1;
    % Two stdev from estimated mean
    f = fill([x, fliplr(x)],[mu+2*sqrt(var), fliplr(mu-2*sqrt(var))],'r','edgecolor','none');
    set(f,'facealpha',0.2);
    % red is estimated mean, green is actual mean
    plot(x,mu,'r',x,y,'g');
    % points are samples
    scatter(xt,t,'black');
    xlabel('x');
    ylabel('y');
    title(['Gaussian Process Regression N = ',num2str(numSamples),' points']);
    hold off;
end
