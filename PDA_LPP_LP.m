% =====================================================================
% Code for the conference paper:
% Qian Wang, Toby Breckon, Source Class Selection with Label Propagation
% for Partial Domain Adaptation, ICIP 2021
% By Qian Wang, qian.wang173@hotmail.com
% =====================================================================
function [acc, acc_per_class] = PDA_LPP_LP(domainS_features,domainS_labels,domainT_features,domainT_labels,d,T, NumNeighbors,useClassMeans)
num_iter = T;
options.ReducedDim = d;
options.alpha = 1;
num_class = length(unique(domainS_labels));
W = zeros(size(domainS_features,1)+size(domainT_features,1));
W_s = constructW(domainS_labels);
W(1:size(W_s,1),1:size(W_s,2)) =  W_s;
% looping
fprintf('d=%d\n',options.ReducedDim);
reservedClasses = ones(1,num_class);
for iter = 1:num_iter
    P = LPP([domainS_features;domainT_features],W,options);
    %P = eye(size(domainS_features,2));
    domainS_proj = domainS_features*P;
    domainT_proj = domainT_features*P;
    proj_mean = mean([domainS_proj;domainT_proj]);
    domainS_proj = domainS_proj - repmat(proj_mean,[size(domainS_proj,1) 1 ]);
    domainT_proj = domainT_proj - repmat(proj_mean,[size(domainT_proj,1) 1 ]);
    domainS_proj = L2Norm(domainS_proj);
    domainT_proj = L2Norm(domainT_proj);
    % pseudo-labeling with label propagation
    if useClassMeans
        classMeans = zeros(num_class,size(domainS_proj,2));
        for i = 1:num_class
            classMeans(i,:) = mean(domainS_proj(domainS_labels==i,:));
        end
        classMeans = L2Norm(classMeans);    
        Mdl = fitsemigraph(classMeans,[1:num_class]',domainT_proj, 'NumNeighbors',NumNeighbors);
    else
        Mdl = fitsemigraph(domainS_proj,domainS_labels',domainT_proj, 'NumNeighbors',NumNeighbors);
    end
    score = zeros(size(domainT_proj,1),num_class);
    score(:,logical(reservedClasses)) = Mdl.LabelScores;
    expMatrix = exp(score);
    probMatrix = expMatrix./repmat(nansum(expMatrix,2),[1 num_class]);
    [prob,predLabels] = max(probMatrix');
    reservedClasses = ones(1,num_class);
    for i = 1:num_class
        if sum(predLabels==i) <= 0
            reservedClasses(i) = 0;
        end
    end
    % select source samples
    reservedSourceSamples = zeros(length(domainS_labels),1);
    reservedSourceSamples = reservedClasses(domainS_labels);
    trustable = reservedClasses(predLabels);
    domainS_features = domainS_features(logical(reservedSourceSamples),:);
    domainS_labels = domainS_labels(logical(reservedSourceSamples));
    pseudoLabels = predLabels;
    pseudoLabels(~trustable) = -1;
    W = constructW([domainS_labels,pseudoLabels]);
    %% calculate ACC
    acc(iter) = sum(predLabels==domainT_labels)/length(domainT_labels);
    for i = 1:num_class
        acc_per_class(iter,i) = nansum((predLabels == domainT_labels).*(domainT_labels==i))/nansum(domainT_labels==i);
    end
    fprintf('Iteration=%d, Acc:%0.3f,Mean acc per class: %0.3f\n', iter, acc(iter), nanmean(acc_per_class(iter,:)));
end
