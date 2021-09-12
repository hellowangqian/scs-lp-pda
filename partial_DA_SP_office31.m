% =====================================================================
% Code for the conference paper:
% Qian Wang, Toby Breckon, Source Class Selection with Label Propagation
% for Partial Domain Adaptation, ICIP 2021
% By Qian Wang, qian.wang173@hotmail.com
% =====================================================================
%% Loading Data:
% Features are extracted using resnet50 pretrained on ImageNet without
% fine-tuning
clear all
addpath('./utils/');
%data_dir = '/mnt/HD2T/DomainAdaptation/Office31data/';
data_dir = 'E:\DomainAdaptation0\Office31data\';
domains = {'A','D','W'};
count = 0;
pcaDim = 0;
lppDim = 128;
T = 10;
numNeighbors = 10;
useClassMeans = 1;

targetClassSelector = [1,2,6,11,12,13,16,17,18,23];
for source_domain_index = 1:length(domains)
    load([data_dir 'office-' domains{source_domain_index} '-resnet50-noft']);
    domainS_features_ori = L2Norm(resnet50_features);
    %domainS_features_ori = resnet50_features;
    domainS_labels = labels+1;
    
    for target_domain_index = 1:length(domains)
        if target_domain_index == source_domain_index
            continue;
        end
        fprintf('Source domain: %s, Target domain: %s\n',domains{source_domain_index},domains{target_domain_index});
        load([data_dir 'office-' domains{target_domain_index} '-resnet50-noft']);
        domainT_features = L2Norm(resnet50_features);
        domainT_features=resnet50_features;
        if pcaDim > 0
            opts.ReducedDim = pcaDim;
            X = double([domainS_features_ori;domainT_features]);
            P_pca = PCA(X,opts);
            domainS_features = domainS_features_ori*P_pca;
            domainT_features = domainT_features*P_pca;
            domainS_features = L2Norm(domainS_features);
        else
            domainS_features = L2Norm(domainS_features_ori);
        end
        domainT_features = L2Norm(domainT_features);
        domainT_labels = labels+1;
        targetInstanceSelector = zeros(1,length(domainT_labels));
        for i = 1:length(domainT_labels)
            targetInstanceSelector(i) = sum(domainT_labels(i) == targetClassSelector);
        end
        targetInstanceSelector = logical(targetInstanceSelector);
        domainT_features = domainT_features(targetInstanceSelector,:);
        domainT_labels = domainT_labels(targetInstanceSelector);
        num_class = length(unique(domainT_labels));
        %% Proposed method:
        [acc,acc_per_class ]= PDA_LPP_LP(domainS_features,domainS_labels,domainT_features,domainT_labels,lppDim,T,numNeighbors,useClassMeans);          
        count = count + 1;
        all_acc_per_class(count,:) = nanmean(acc_per_class,2);
        all_acc_per_image(count,:) = acc;
    end
end
mean_acc_per_class = nanmean(all_acc_per_class,1)
mean_acc_per_image = nanmean(all_acc_per_image,1)
save(['office31-PDA-useClassMeans-' num2str(useClassMeans) '-numNeighbors-' num2str(numNeighbors) '-PcaDim-' num2str(pcaDim) '-LppDim-' num2str(lppDim) '-T-' num2str(T) '.mat'],'all_acc_per_class','all_acc_per_image','mean_acc_per_class','mean_acc_per_image');
