%%% ECE 414 Project 1
%%% Eugene Sokolov, Sheryan Resutov, Harrison Zhao

clear all
close all 
clc

%%% For Gaussian RV with unknown mean and variance
mu = rand(1)
var = rand(1)
iterations = 10000;
numsamp = 100;
gmuN = zeros(numsamp,1);
gmse = zeros(numsamp,1);

hyp = [0;0.5;1;2;3;4;8;15];
hypSize = size(hyp,1);
gconj = zeros(hypSize,numsamp);

for ii = 1:iterations
 gaus = normrnd(mu,sqrt(var),numsamp,1);
 mse = zeros(numsamp,1);
 muN = zeros(numsamp,1);
 muN(1) = gaus(1);

 conj = zeros(hypSize,numsamp);
 
 for N = 2:numsamp

   for hypI = 2:hypSize
    
    a=1 * 1/(N*hyp(hypI,1) + 1);
    b=muN(N)  * (N*hyp(hypI,1))/(N*hyp(hypI,1) + 1);
    conj(hypI,N) = a+b;
   end

   err = (gaus(N) - muN(N-1));
   muN(N) = muN(N-1) + err/N;
   mse(N) = err^2;

 end 
 gmuN = gmuN + muN;
 gmse = gmse + mse;
 gconj = gconj + conj;
end
gmuN = gmuN/iterations;
gmse = gmse/iterations;
gconj = gconj/iterations;

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


%%% For Binomial RV with unknown mean and variance
n = randi(10)
p = rand(1)
iterations = 10000;
numsamp = 100;
gmuN2 = zeros(numsamp,1);
gmse2 = zeros(numsamp,1);

mubin = n*p
varbin = n*p*(1-p)
hyp = [1 1;2 2;3 3;4 4;5 5;4 8;1 8;8 4;8 1];
hypSize = size(hyp,1);
gconj2 = zeros(hypSize,numsamp);

for ii = 1:iterations
 bino = binornd(n,p,numsamp,1);
 mse2 = zeros(numsamp,1);
 muN2 = zeros(numsamp,1);
 muN2(1) = bino(1);
 c = bino(1);
 conj2 = zeros(hypSize, numsamp);
 
 for N = 2:numsamp
 
   for hypI = 2:hypSize
    %(M+a)/(M+a+l+b)
    conj2(hypI,N) = (c+hyp(hypI,1))/(c + hyp(hypI,1) + (N*n-c) +hyp(hypI,2)); 
   end

   err2 = (bino(N) - c/(N-1));
   c=c+bino(N);
   muN2(N) = c/N;
   mse2(N) = err2^2;

 end 
 gmuN2 = gmuN2 + muN2;
 gmse2 = gmse2 + mse2;
 gconj2 = gconj2 + conj2;
end
gmuN2 = gmuN2/iterations;
gmse2 = gmse2/iterations;
gconj2 = gconj2/iterations;

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
plot(conj2')
title('Conjugate Prior for Binomial','FontName','Times')
xlabel('Numbers of Measurements','FontName','Times')
ylabel('P(mu|D)','FontName','Times')
hleg = legend('Binomial');


