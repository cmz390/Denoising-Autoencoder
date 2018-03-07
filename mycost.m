function perf = mycost(net, varargin)
%MSESPARSE Mean squared error performance function with L2 and sparsity
%regularizers.
%
% <a href="matlab:doc mse">msesparse</a>(net,targets,outputs,errorWeights,...parameters...) calculates a
% network performance given targets, outputs, error weights and parameters
% as the mean of squared errors.
%
% Only the first four arguments are required.  The default error weight
% is {1}, which weights the importance of all targets equally.
%
% Parameters are supplied as parameter name and value pairs:
%
% 'regularization' - a fraction between 0 (the default) and 1 indicating
%   the proportion of performance attributed to weight/bias values. The
%   larger this value the network will be penalized for large weights, and 
%   the more likely the network function will avoid overfitting.
%
% 'normalization' - this can be 'none' (the default), or 'standard', which
%   results in outputs and targets being normalized to [-1, +1], and
%   therefore errors in the range [-2, +2), or 'percent' which normalizes
%   outputs and targets to [-0.5, 0.5] and errors to [-1, 1].
%
% 'L2Regularization' - This parameter controls the weighting of an L2
%   regularizer for the weights of the network (and not the biases).
%
% 'sparsityRegularization' - This parameter controls the weighting of a
%   sparsity regularizer, which discourages large fractions of the neurons 
%   in the first layer from activating in response to an input.
%
% 'sparsity' - This parameter controls the desired fraction of neurons that
%   should activate in the first layer in response to an input. This must 
%   be between 0 and 1.
%
% See also MSE, SSE, MAE.

% Copyright 2014-2015 The MathWorks, Inc.

% Function Info
persistent INFO;
if isempty(INFO), INFO = nnModuleInfo(mfilename); end
if nargin == 0, perf = INFO; return; end

% NNET Backward Compatibility
% WARNING - This functionality may be removed in future versions
if ischar(net) && strcmp(net,'info')
  perf = INFO; return
elseif ischar(net) || ~(isa(net,'network') || isstruct(net))
  perf = nnet7.performance_fcn(mfilename,net,varargin{:}); return
end

% Arguments
param = nn_modular_fcn.parameter_defaults(mfilename);
[args,param,nargs] = nnparam.extract_param(varargin,param);
if (nargs < 2), error(message('nnet:Args:NotEnough')); end
t = args{1};
y = args{2};
if nargs < 3, ew = {1}; else ew = varargin{3}; end
net.performFcn = mfilename;
net.performParam = param;

% Apply
perf = nncalc.perform(net,t,y,ew,param);