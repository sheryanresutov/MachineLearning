%%% ECE 414 Project 1
%%% Eugene Sokolov, Sheryan Resutov, Harrison Zhao

clear all
close all 
clc

%%% For Gaussian RV with unknown mean and variance
mu = rand(1);
var = rand(1);
iterations = 10000;
numsamp = 100;
gmuN = zeros(numsamp,1);
gmse = zeros(numsamp,1);

hyp = zeros(numsamp, 4); % mu, lambda, alpha, beta
mu0 = 0;
lambda = 5;
alpha = 5;
beta = 5;
initparams = [mu0, lambda, alpha, beta];

mypdf = @(mu0, lambda, a, b, x, t) (b.^a.*sqrt(lambda))./(gamma(a).*sqrt(2.*pi)).*...
         t.^(a-.5).*exp(-b.*t).*exp((-lambda.*t.*(x-mu0).^2)/2);

for ii = 1:iterations
    gaus = normrnd(mu,sqrt(var),numsamp,1);
    mse = zeros(numsamp,1);
    muN = zeros(numsamp,1);
    muN(1) = gaus(1);
    localhyp = zeros(numsamp, 4);
    localhyp(1,:) = initparams;
    cumulativeSum = 0;
    
    for N = 2:numsamp       
        err = gaus(N) - muN(N-1);
        muN(N) = muN(N-1) + err/N;
        mse(N) = err^2;
        cumulativeSum = cumulativeSum + mse(N);
        
        localhyp(N, 1) = (lambda * mu0 + N * muN(N))/(lambda + N);
        localhyp(N, 2) = lambda + N;
        localhyp(N, 3) = alpha + N/2;
        localhyp(N, 4) = beta + (0.5 * cumulativeSum) + ...
            (N * lambda * (muN(N) - mu0)^2)/2 * (lambda + N);
    end
    gmuN = gmuN + muN;
    gmse = gmse + mse;
    hyp = hyp + localhyp;
end
gmuN = gmuN/iterations;
gmse = gmse/iterations;
hyp = hyp/iterations;

%Plot sequential MLE, conjugate prior estimate
figure 
plot(gmuN,'b*')
title('Max Likelihood Estimate vs Numbers of Measurements','FontName','Times')
xlabel('Numbers of Measurements','FontName','Times')
ylabel('Mean','FontName','Times')
hleg = legend('Gaussian'); 

%Plot MSE
figure 
plot(gmse,'b*')
title('MSE vs Numbers of Measurements','FontName','Times')
xlabel('Numbers of Measurements','FontName','Times')
ylabel('MSE','FontName','Times')
hleg = legend('Gaussian');

%Plot conjugate priors
X = -10:.05:10;
T = 0:.005:.1;
[x,t] = meshgrid(X,T);
figure;
surf(x,t,mypdf(hyp(1,1), hyp(1,2), hyp(1,3), hyp(1,4), x, t));
title('Initial Conjugate Prior for Gaussian','FontName','Times');

figure;
surf(x,t,mypdf(hyp(numsamp/4,1), hyp(numsamp/4,2), hyp(numsamp/4,3), hyp(numsamp/4,4), x, t));
title('Numsamples/4 Conjugate Prior for Binomial','FontName','Times');

figure;
surf(x,t,mypdf(hyp(numsamp/2,1), hyp(numsamp/2,2), hyp(numsamp/2,3), hyp(numsamp/2,4), x, t));
title('Numsamples/2 Conjugate Prior for Binomial','FontName','Times');

figure;
surf(x,t,mypdf(hyp(numsamp,1), hyp(numsamp,2), hyp(numsamp,3), hyp(numsamp,4), x, t));
title('Final Conjugate Prior for Gaussian','FontName','Times');

%% For Binomial RV with unknown mean and variance
n = randi(10)
p = rand(1)
iterations = 10000;
numsamp = 100;
gmuN2 = zeros(numsamp,1);
gmse2 = zeros(numsamp,1);

mubin = n*p
varbin = n*p*(1-p)
hyp = zeros(numsamp, 2); %alpha, beta
% hyper parameters
alpha = 1;
beta = 1;

for ii = 1:iterations
    bino = binornd(n,p,numsamp,1);
    mse2 = zeros(numsamp,1);
    muN2 = zeros(numsamp,1);
    muN2(1) = bino(1);
    c = bino(1);
    conj2 = zeros(hypSize, numsamp);
    localhyp = zeros(numsamp, 2);
 
    for N = 2:numsamp
        err2 = (bino(N) - c/(N-1));
        c=c+bino(N);
        muN2(N) = c/N;
        mse2(N) = err2^2;
        localhyp(N, 1) = alpha + c;
        localhyp(N, 2) = beta + N*n-c;
    end 
    gmuN2 = gmuN2 + muN2;
    gmse2 = gmse2 + mse2;
    hyp = hyp + localhyp;
end
gmuN2 = gmuN2/iterations;
gmse2 = gmse2/iterations;
hyp = hyp/iterations;

%Plot sequential MLE
figure 
plot(gmuN2,'r*')
title('Max Likelihood Estimate vs Numbers of Measurements','FontName','Times')
xlabel('Numbers of Measurements','FontName','Times')
ylabel('Mean','FontName','Times')
hleg = legend('Binomial'); 

%Plot MSE
figure 
plot(gmse2,'r*')
title('MSE vs Numbers of Measurements','FontName','Times')
xlabel('Numbers of Measurements','FontName','Times')
ylabel('MSE','FontName','Times')
hleg = legend('Binomial');

%Plot conjugate prior
figure
subplot(2, 2, 1);
plot(0:0.01:1, pdf('beta', 0:0.01:1, hyp(2, 1), hyp(2, 2)));
title('Initial Conjugate Prior for Binomial','FontName','Times')
subplot(2, 2, 2);
plot(0:0.01:1, pdf('beta', 0:0.01:1, hyp(numsamp/4, 1), hyp(numsamp/4, 2)));
title('Numsamples/4 Conjugate Prior for Binomial','FontName','Times')
subplot(2, 2, 3);
plot(0:0.01:1, pdf('beta', 0:0.01:1, hyp(numsamp/2, 1), hyp(numsamp/2, 2)));
title('Numsamples/2 Conjugate Prior for Binomial','FontName','Times')
subplot(2, 2, 4);
plot(0:0.01:1, pdf('beta', 0:0.01:1, hyp(numsamp, 1), hyp(numsamp, 2)));
title('Final Conjugate Prior for Binomial','FontName','Times')
hleg = legend('Binomial');


