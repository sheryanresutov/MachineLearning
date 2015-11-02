function [ mus, pCs ] = binGaussianGenerativeModel(x, t)
    %C1 tn = 1, C2 tn = 0
    [nObservations] = size(x, 1);
    classCounts = [sum(t==1) sum(t==0)];
    classCounts
    pCs = classCounts / nObservations;
    
    mus = zeros(2); % 2 x 2
    for ii=1:2
        tt = repmat(t,1,2);
        if ii==2
            tt = 1 - tt;
        end
        tt
        sum(tt .* x, 1)
        mus(ii, :) = sum(tt .* x, 1) / classCounts(ii);
    end
%     as = cell(nFeatures, 1);
%     for ii=1:nFeatures
%         f = @(x) sum(x .* log(mus(ii,:)) + (1 - x).*log(1-mus(ii,:))) + log(pCs(ii));
%     end
end
