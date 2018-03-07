% %my denosing autoencoder
% clear; close all; clc;
% % Load the training data into memory
% [xTrainImages,tTrain] = digitTrainCellArrayData;
% 
% % Display some of the training images
% clf
% 
% for i = 1:20
%     subplot(4,5,i);
%     imshow(xTrainImages{i});
% end
% 
% J  = {};
% for i = 1:length(xTrainImages)
%  J{i} = imnoise(xTrainImages{i},'gaussian',0, 0.1);
% end
% 
%  hold on;
%  figure;
% % imshow(J);
% 
% for i = 1:20
%     subplot(4,5,i);
%     imshow(J{i});
% end
% 
% [lx, ly] = size(xTrainImages{1});
% 
% hiddenSize = 5 * lx * ly;
% a = 0.0001;
% b = 0.01;
% p = 0.05;
% 
% 
% rng('default')
% 
% all = {[J,xTrainImages]};
% all = all{1};
% %pre-training
% 
% autoenc0 = mytrainAutoencoder(all,hiddenSize, ...
%     'MaxEpochs',1, ...
%     'L2WeightRegularization',a, ...
%     'SparsityRegularization',b, ...
%     'SparsityProportion',p, ...
%     'ScaleData', false, ... 
%     'UseGPU', false);
% 
% view(autoenc0);
% 
% 
% %fine tuning
% for i = 1:numel(all)
%     xTrain(:,i) = all{i}(:);
% end
% 
% % Perform fine tuning
% % autoenc0 = train(autoenc0,xTrain);
% 
% 
% % get the next input data h(x) and h(y)
feat0 = encode(autoenc0, all);

% 
% % deep learning
%  autoenc1 = trainAutoencoder(feat0,hiddenSize, ...
%      'MaxEpochs',500, ...
%      'L2WeightRegularization',a, ...
%      'SparsityRegularization',0, ...
%      'SparsityProportion',p, ...
%       'ScaleData', false, ...
%   'UseGPU', true);
%  
%   feat1 = encode(autoenc1,feat0);
%   
   autoenc1 = trainAutoencoder(feat0,hiddenSize, ...
     'MaxEpochs',1, ...
     'L2WeightRegularization',a, ...
     'SparsityRegularization',0, ...
     'SparsityProportion',p, ...
      'ScaleData', false, ...
      'UseGPU', true);
% 

view(autoenc1);
deepnet = stack(autoenc0,autoenc1);
% 
view(deepnet);
% 
% %fine tuning
% % for i = 1:numel(feat0)
% %     yTrain(:,i) = feat0{i}(:);
% % end
% 
% % Perform fine tuning
% deepnet = train(deepnet,feat0);
% 
% final0 = deepnet(feat0);
% 
% 
final = decode(autoenc0,feat0);

%Display some result images
hold on;
figure;
for i = 1:20
    subplot(4,5,i);
    imshow(final{i});
end