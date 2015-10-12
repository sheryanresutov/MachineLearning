%%% ECE 414 Project 2
%%% Eugene Sokolov, Sheryan Resutov, Harrison Zhao

clear all
close all
clc

%% Part 1
a0 = -0.3;
a1 = 0.5;
sigmaN = 0.2;
alpha = 2;
beta = 25;
numSamp = 20;
nLines = 6;

xSpace=linspace(-1,1,100);

mu0 = 0;
mu1 = 0;
sigma0 = 1;
sigma1 = 1;

xsweep = linspace(-1,1,100);

x = unifrnd(-1,1,1,numSamp);
f = a0 + a1*x;
noise = normrnd(0,sigmaN,1,numSamp);
t = f + noise;
phi = [];
y = ones(1,numSamp);

[W0,W1] = meshgrid(linspace(-1,1,1000)', linspace(-1,1,1000)');
W = [W0(:) W1(:)];
Wmean = [mu0 mu1];
Wstdev = [sigma0 sigma1];
p = mvnpdf(W, Wmean, Wstdev);
subplot(1,3,2)
surf(W0,W1,reshape(p,vals,vals),'edgecolor', 'none');
view(2)
title('before any data points')

w0 = normrnd(mu0,sigma0,1,nLines);
w1 = normrnd(mu1,sigma1,1,nLines);
    
    for ii = 1:6
            y = w0(ii) + w1(ii) * xsweep;
            subplot(1,3,3)
            hold on
            plot(xsweep,y,'r');
            xlabel('x')
            ylabel('y')
	end


for N = 1:numSamp
    
    phi = [phi; -x(N)+1 x(N)+1];        
    A=[1 1;-1 1];
    Sinv = alpha*eye(2) + beta*phi'*phi;
    Sn = inv(Sinv);
    mN = beta*Sn*phi'*t(1:N)';
    
    D = A*mN;
    mu0 = D(1);
    mu1 = D(2);
    sigma0 = Sn(1,1);
    sigma1 = Sn(2,2);
    w0 = normrnd(mu0,sigma0,1,nLines);
    w1 = normrnd(mu1,sigma1,1,nLines);
    
    likelihood = normpdf(t(N),W0+W1*x(N),1/beta);
    
    if(N==1||N==2||N==20)
        figure
        hold on
        subplot(1,3,1)
        surf(W0,W1,likelihood,'edgecolor','none');
        view(2)

        plot3(a0,a1,100,'w+')
        subplot(1,3,2)
        p = mvnpdf(W, Wmean, Wstdev);
        surf(W0,W1,reshape(p,1000,1000),'edgecolor', 'none');
        view(2)

        xlabel('w0')
        ylabel('w1')
        title(['Observing with ',num2str(N),' data points'])
        
        for ii = 1:6
            y = w0(ii) + w1(ii) * xsweep;
            subplot(1,3,3)
            hold on
            plot(xsweep,y,'r');
            xlabel('x')
            ylabel('y')
        end
        plot(x(1:N),t(1:N),'bo');
    end
    
end

% nLines = 6;
% 
% w0 = normrnd(0,1,1,nLines);
% w1 = normrnd(0,1,1,nLines);
% wMean = [0,0];
% wSigma = [1,1];
% 
% 
% 
% x = linspace(-1,1,numSamp);
% y = w0 + w1*x;
% 
% for i = 1:nLines
% 	subplot(1,3,3)
% 	plot(x,y,'r')
% 	ylim([-1 1])
% end



%% Part 2
alpha = 2;
beta = 25;
iter = 100;
numSamp = 25;
sigmaN = 0.2;
nBases = 9;
i = 1;

x = linspace(0,1,iter);
actual = sin(2*pi*x);
X = rand(1,numSamp);
noise = normrnd(0,sigmaN,1,numSamp);
t = sin(2*pi*X) + noise;

PHI= [];
p = zeros(iter,9);
phi = @(x,j) exp(-(x - 0.1*j).^2/sigmaN^2);
for j = 1:9
    p(:,j) = phi(x,j)';
end

for N = 1:numSamp
    
    PHI = [PHI; phi(X(N),1:9)];        
    Sinv = alpha*eye(nBases) + beta*PHI'*PHI;
    Sn = inv(Sinv);
    mN = beta*Sn*PHI'*t(1:N)';
    
    pred = (p * mN)';
    v = diag(repmat(1/beta,iter,iter) + p*Sn*p')';
    topPred = pred + sqrt(v);
    botPred = pred - sqrt(v);
    %v = repmat(1/beta,iter,iter) + p*Sn*p';
    %topPred = pred + sqrt(v(:,N))';
    %botPred = pred - sqrt(v(:,N))';

	if(N==1||N==2||N==4||N==numSamp)
	
	subplot(2,2,i)
    hold on;
	plot(x,actual,'g');
	plot(x,pred,'r');
    plot(X(1:N), t(1:N), 'bo');
    X1=[x,fliplr(x)];
    Y=[topPred,fliplr(botPred)];
    h = fill(X1,Y,'r','edgecolor','none');
    set(h,'facealpha',0.2);
        
	
	title(['predictive distribution with ',num2str(N),' points']);
	xlabel('x');
	ylabel('t');	
    hold off;
    i=i+1;
	end
end



