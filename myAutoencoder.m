classdef(Sealed, InferiorClasses = {?network}) myAutoencoder
    % AUTOENCODER   Autoencoder
    %   An autoencoder is a type of neural network which consists of an
    %   encoder and a decoder. The encoder maps the input to a hidden
    %   representation, and the decoder attempts to map this hidden
    %   representation back to the original input.
    %
    %   Autoencoder properties:
    %       HiddenSize              - Size of the hidden representation
    %       EncoderTransferFunction - The transfer function for the encoder
    %       EncoderWeights          - The weights of the encoder
    %       EncoderBiases           - The bias vector for the encoder
    %       DecoderTransferFunction - The transfer function for the decoder
    %       DecoderWeights          - The weights of the decoder
    %       DecoderBiases           - The bias vector for the decoder
    %       TrainingParameters      - A structure that holds the training
    %                                 parameters for the autoencoder
    %       ScaleData               - True when the input data is rescaled
    %
    %   Autoencoder methods:
    %       network                 - Convert the autoencoder into a
    %                                 network object
    %       encode                  - Encode input data
    %       decode                  - Decode encoded data
    %       generateFunction        - Generate a MATLAB function for
    %                                 running the autoencoder
    %       generateSimulink        - Generate a simulink model for the
    %                                 autoencoder
    %       predict                 - Run the autoencoder on some inputs
    %       view                    - View a diagram of the autoencoder
    %       stack                   - Stack the encoders from several
    %                                 autoencoders together
    %       plotWeights             - Plot a visualization of the weights
    %                                 for the encoder of an autoencoder
    %
    %   Example:
    %       Train a sparse autoencoder on images of handwritten digits to
    %       learn features, and use it to compress and reconstruct these
    %       images. View some of the original images along with their
    %       reconstructed versions.
    %
    %       x = digitSmallCellArrayData;
    %       hiddenSize = 40;
    %       autoenc = trainAutoencoder(x, hiddenSize, ...
    %           'L2WeightRegularization', 0.004, ...
    %           'SparsityRegularization', 4, ...
    %           'SparsityProportion', 0.15);
    %       xReconstructed = predict(autoenc, x);
    %       figure;
    %       for i = 1:20
    %           subplot(4,5,i);
    %           imshow(x{i});
    %       end
    %       figure;
    %       for i = 1:20
    %           subplot(4,5,i);
    %           imshow(xReconstructed{i});
    %       end
    %
    %   See also trainAutoencoder
    
    % Copyright 2015-2016 The MathWorks, Inc.
    
    properties(Access = private)
        Version = 1
        Network
        VisualizationDimensions
        PrivateUseGPU
        TrainedOnImages
    end
    
    properties(SetAccess = private, Dependent)
        % HiddenSize   Size of hidden representation
        %   The HiddenSize is the size of the representation in the hidden
        %   layer of the autoencoder.
        HiddenSize
        
        % EncoderTransferFunction   The transfer function for the encoder
        %   EncoderTransferFunction is the name of the transfer function
        %   that is used for the encoder.
        EncoderTransferFunction
        
        % EncoderWeights   The weights of the encoder
        %   EncoderWeights is a matrix of the weight values for the
        %   encoder.
        EncoderWeights
        
        % EncoderBiases   The bias vector for the encoder
        %   EncoderBiases is a vector of bias values for the encoder.
        EncoderBiases
        
        % DecoderTransferFunction   The transfer function for the decoder
        %   DecoderTransferFunction is the name of the transfer function
        %   that is used for the decoder.
        DecoderTransferFunction
        
        % DecoderWeights   The weights of the decoder
        %   DecoderWeights is a matrix of the weight values for the
        %   decoder.
        DecoderWeights
        
        % DecoderBiases   The bias vector for the decoder
        %   DecoderBiases is a vector of bias values for the decoder.
        DecoderBiases
        
        % TrainingParameters   A structure containing the parameters that were used for training
        %   TrainingParameters is a structure that holds the parameters
        %   which were used to train the Autoencoder.
        TrainingParameters
        
        % ScaleData   True when the input data is rescaled to the same range as the output layer
        %   ScaleData is a logical value which is used to indicate whether
        %   any data passed to the autoencoder should be rescaled.
        %   Autoencoders attempt to replicate their input at their output.
        %   For this to be possible, the range of the input data must match
        %   the range of the transfer function for the decoder. When
        %   ScaleData is true, when training an autoencoder the training
        %   data will be automatically scaled to this range. Scaling will
        %   also take place when calling predict, encode and decode. The
        %   default value is true.
        ScaleData
    end
    
    methods
        function val = get.HiddenSize(this)
            val = this.Network.layers{1}.size;
        end
        
        function val = get.EncoderTransferFunction(this)
            val = this.Network.layers{1}.transferFcn;
        end
        
        function val = get.EncoderWeights(this)
            val = this.Network.IW{1};
        end
        
        function val = get.EncoderBiases(this)
            val = this.Network.b{1};
        end
        
        function val = get.DecoderTransferFunction(this)
            val = this.Network.layers{2}.transferFcn;
        end
        
        function val = get.DecoderWeights(this)
            val = this.Network.LW{2,1};
        end
        
        function val = get.DecoderBiases(this)
            val = this.Network.b{2};
        end
        
        function val = get.TrainingParameters(this)
            val = struct( ...
                'DataDivision', this.Network.divideFcn, ...
                'LossFunction', this.Network.performFcn, ...
                'Algorithm', this.Network.trainFcn, ...
                'MaxEpochs', this.Network.trainParam.epochs, ...
                'L2WeightRegularization', this.Network.performParam.L2WeightRegularization, ...
                'SparsityRegularization', this.Network.performParam.sparsityRegularization, ...
                'SparsityProportion', this.Network.performParam.sparsity, ...
                'ShowProgressWindow', this.Network.trainParam.showWindow, ...
                'UseGPU', this.PrivateUseGPU ...
                );
        end
        
        function val = get.ScaleData(this)
            val = iNetworkUsesMapMinMaxAtInput(this.Network);
        end
        
        function net = network(this)
            % network   Create a network object from this autoencoder
            %   net = network(autoenc) returns a network that is equivalent
            %   to the autoencoder, autoenc.
            %
            %   See also network
            net = this.Network;
        end
        
        function z = encode(this, x)
            % encode   Encode input data
            %   z = encode(autoenc, x) returns the encoded data z for the
            %   input data x, using the autoencoder autoenc. The encoded
            %   data z is the result of passing x through the encoder of
            %   autoenc.
            %
            %   If autoenc was trained on a cell array of images, then x
            %   must either be a cell array of images or a matrix of single
            %   image data. If autoenc was trained on a matrix where each
            %   column represents a single sample, then x must be a matrix
            %   which also has this format.
            %
            %   See also Autoencoder.decode
            encoder = this.getEncoder();
            x = this.convertDataIntoAMatrixOfColumns(x);
            z = encoder(x);
        end
        
        function y = decode(this, z)
            % decode   Decode encoded data
            %   y = decode(autoenc, z) returns the decoded data y for the
            %   encoded data z, using the autoencoder autoenc. The decoded
            %   data y is the result of passing z through the decoder of
            %   autoenc.
            %
            %   The encoded data z must be a matrix where each column
            %   represents a single encoded sample. If autoenc was trained
            %   on a cell array of images, then y will also be a cell array
            %   of images. If autoenc was trained on a matrix where each
            %   column represents a single sample, then y will also be a
            %   matrix where each column represents a single sample.
            %
            %   See also Autoencoder.encode
            if(~iIsNumericAndRealMatrixWithThisNumberOfRows(z, this.HiddenSize))
                error(message('nnet:autoencoder:InvalidDataForDecode', inputname(1)));
            end
            decoder = this.getDecoder();
            y = decoder(z);
            
            if(this.TrainedOnImages)
                y = iFormatMatrixOfImagesIntoCellArray(y, this.VisualizationDimensions);
            end
        end
        
        function generateFunction(this, varargin)
            % generateFunction   Generate a MATLAB function for running the autoencoder
            %   generateFunction(autoenc) generates a complete stand-alone
            %   MATLAB function for running the autoencoder on input data
            %   in the current directory.
            %
            %   generateFunction(autoenc, pathname) generates a complete
            %   stand-alone MATLAB function for running the autoencoder on
            %   input data in the location specified by pathname.
            %
            %   generateFunction(..., Name, Value) generates a function
            %   with additional options specified by the following
            %   name/value pairs:
            %
            %       'ShowLinks'     - A logical value which determines
            %                         whether links to the generated code
            %                         should be displayed in the command
            %                         window. The default value is true.
            %
            %   See also network.genFunction
            
            %       'MatrixOnly'    - A logical value which determines
            %                         whether the generated code should
            %                         only use matrices to make it
            %                         compatible with MATLAB coder. From
            %                         R2016b onwards, only the value true
            %                         is allowed.
            
            p = inputParser;
            
            defaultPathname = 'neural_function';
            defaultShowLinks = true;
            defaultMatrixOnly = true;
            
            p.addOptional('pathname', defaultPathname, @iValidateCharacterArray)
            p.addParameter('ShowLinks', defaultShowLinks, @iValidateScalarLogical);
            p.addParameter('MatrixOnly', defaultMatrixOnly);
            
            p.parse(varargin{:});
            
            iThrowErrorForInvalidMatrixOnlyValue(p.Results.MatrixOnly);
            iIssueWarningIfUserSetMatrixOnlyToTrue(p);
            iCallGenFunctionWithCorrectArguments(this.Network, p.Results);
        end
        
        function generateSimulink(this)
            % generateSimulink   Generate a simulink model for the autoencoder
            %   generateSimulink(autoenc) creates a Simulink model for the
            %   autoencoder autoenc.
            %
            %   See also network.gensim
            gensim(this.Network);
        end
        
        function y = predict(this, x)
            % predict   Run an autoencoder on a set of inputs
            %   y = predict(autoenc, x) returns the predictions y for the
            %   input data x, using the autoencoder autoenc. The result y
            %   will be a reconstruction of x.
            %
            %   If autoenc was trained on a cell array of images, then x
            %   must be a cell array of images or a matrix of single image
            %   data. If autoenc was trained on a matrix where each column
            %   is a sample, then x must be a matrix where each column is a
            %   sample. The output y that is returned will be in the same
            %   format as x.
            %
            %   See also Autoencoder.encode, Autoencoder.decode
            xColumns = this.convertDataIntoAMatrixOfColumns(x);
            y = sim(this.Network, xColumns);
            y = this.convertMatrixOfColumnsIntoData(x, y);
        end
        
        function view(this)
            % view   View a diagram of the autoencoder
            %   view(autoenc) generates a diagram of the autoencoder
            %   autoenc.
            %
            %   See also network.view
            view(this.Network);
        end
        
        function stackednet = stack(varargin)
            % stack   Stack the encoders from several autoencoders together
            %   stackednet = stack(autoenc1, autoenc2, ...) will take a
            %   series of autoencoders, and return a network object that is
            %   created by stacking the encoders of these autoencoders
            %   together. The autoencoders will be stacked from left to
            %   right, so that the first argument will be at the input of
            %   the stacked network, and the last argument will be at the
            %   stacked network's output.
            %
            %   stackednet = stack(autoenc1, autoenc2, ..., net1) will take
            %   a series of autoencoder objects and a final argument which
            %   is a network object, and will return a stacked network,
            %   which is formed by stacking the encoders from the
            %   autoencoders together with the network. Input arguments are
            %   stacked from left to right, so that the first argument will
            %   be at the input of the stacked network, and the last
            %   argument will be at the stacked network's output. The
            %   returned network object stackednet will inherit its
            %   training parameters from the final input argument net1.
            %
            %   The autoencoders and network object can only be stacked
            %   together if their dimensions match. For each autoencoder,
            %   the size of its hidden representation must match the input
            %   size of the next autoencoder or network in the stack.
            %
            %   Example:
            %       Train a sparse autoencoder on glass data and use it to
            %       extract features. Use the features to train a softmax
            %       classifier. Stack the encoder from the autoencoder and
            %       the softmax layer together.
            %
            %       [x,t] = glass_dataset;
            %       hiddenSize = 5;
            %       autoenc = trainAutoencoder(x, hiddenSize, ...
            %           'L2WeightRegularization', 0.001, ...
            %           'SparsityRegularization', 4, ...
            %           'SparsityProportion', 0.05, ...
            %           'DecoderTransferFunction','purelin');
            %       features = encode(autoenc,x);
            %       softnet = trainSoftmaxLayer(features,t);
            %       stackednet = stack(autoenc, softnet);
            %       view(autoenc);
            %       view(softnet);
            %       view(stackednet);
            %
            %   See also Autoencoder
            
            narginchk( 2, Inf );
            iAssertAllAutoencoderOrNetwork(varargin{:});
            
            for i = 1:nargin
                if iIsAutoencoder(varargin{i})
                    varargin{i} = varargin{i}.getEncoder();
                end
            end
            
            stackednet = stack(varargin{:});
        end
        
        function h = plotWeights(this)
            % plotWeights   Plot a visualization of the weights for the encoder of an autoencoder
            %   h = plotWeights(autoenc) takes an autoencoder, autoenc,
            %   and plots a visualization of the encoder weights. This is
            %   useful when an autoencoder has been trained on images, as
            %   it allows the learned features to be visualized. If the
            %   autoencoder has been trained on images, then the
            %   visualization of the weights will have the same dimensions
            %   as the images used for training. The function returns the
            %   handle to the image.
            %
            %   Example:
            %       Train a sparse autoencoder on images of handwritten
            %       digits and visualize the learned features.
            %
            %       x = digitSmallCellArrayData;
            %       hiddenSize = 25;
            %       autoenc = trainAutoencoder(x, hiddenSize, ...
            %           'L2WeightRegularization', 0.004, ...
            %           'SparsityRegularization', 4, ...
            %           'SparsityProportion', 0.2);
            %       plotWeights(autoenc);
            
            firstLayerWeights = this.EncoderWeights;
            numWeightVectors = size(firstLayerWeights, 1);
            imageHeight = this.VisualizationDimensions(1);
            imageWidth = this.VisualizationDimensions(2);
            numImageChannels = iGetNumberOfImageChannels(this.VisualizationDimensions);
            maxValue = max(firstLayerWeights(:));
            
            [numVerticalImages, numHorizontalImages] = iCalculateGalleryDimensions( ...
                imageHeight, imageWidth, numWeightVectors);
            galleryHeight = (imageHeight+1)*numVerticalImages - 1;
            galleryWidth = (imageWidth+1)*numHorizontalImages - 1;
            
            if(numImageChannels == 3)
                imageToShow = repmat(maxValue, galleryHeight, galleryWidth, 3);
            else
                imageToShow = repmat(maxValue, galleryHeight, galleryWidth);
            end
            
            [y, x] = ind2sub([numVerticalImages numHorizontalImages], 1:numWeightVectors);
            for i = 1:numWeightVectors
                startY = (y(i)-1)*(imageHeight+1)+1;
                endY = startY + imageHeight - 1;
                startX = (x(i)-1)*(imageWidth+1)+1;
                endX = startX + imageWidth - 1;
                if(numImageChannels == 3)
                    imageToShow(startY:endY , startX:endX, :) = reshape(firstLayerWeights(i,:)', imageHeight, imageWidth, 3);
                else
                    imageToShow(startY:endY , startX:endX) = reshape(firstLayerWeights(i,:)', imageHeight, imageWidth);
                end
            end
            
            if(numImageChannels == 3)
                for i = 1:3
                    weightImageChannel = imageToShow(:,:,i);
                    imageToShow(:,:,i) = imageToShow(:,:,i) - min(weightImageChannel(:));
                    imageToShow(:,:,i) = imageToShow(:,:,i)./max(weightImageChannel(:));
                end
            else
                imageToShow = imageToShow - min(imageToShow(:));
                imageToShow = imageToShow./max(imageToShow(:));
            end
            
            h = imshow(imageToShow,'InitialMagnification','fit');
        end
    end
    
    methods(Hidden)
        function autoenc = Autoencoder(net, useGPU, visualizationDimensions, trainedOnImages)
            autoenc.Network = net;
            autoenc.PrivateUseGPU = useGPU;
            autoenc.VisualizationDimensions = visualizationDimensions;
            autoenc.TrainedOnImages = trainedOnImages;
        end
    end
    
    methods(Static, Hidden)
        function inputParametersStruct = parseInputArguments(varargin)
            
            p = inputParser;
            
            defaultHiddenSize = 10;
            defaultEncoderTransferFunction = 'logsig';
            validEncoderTransferFunctions = {'logsig','satlin'};
            defaultDecoderTransferFunction = 'logsig';
            validDecoderTransferFunctions = {'logsig','satlin','purelin'};
            defaultLossFunction = 'msesparse';
            validLossFunctions = {'msesparse'};
            defaultTrainingAlgorithm = 'trainscg';
            validTrainingAlgorithms = {'trainscg'};
            defaultMaxEpochs = 1000;
            defaultShowProgressWindow = true;
            defaultL2WeightRegularization = 0.001;
            defaultSparsityRegularization = 1;
            defaultSparsityProportion = 0.05;
            defaultScaleData = true;
            defaultUseGPU = false;
            
            addOptional(p, 'HiddenSize', defaultHiddenSize);
            addParameter(p, 'EncoderTransferFunction', defaultEncoderTransferFunction);
            addParameter(p, 'DecoderTransferFunction', defaultDecoderTransferFunction);
            addParameter(p, 'LossFunction', defaultLossFunction);
            addParameter(p, 'TrainingAlgorithm', defaultTrainingAlgorithm);
            addParameter(p, 'MaxEpochs', defaultMaxEpochs);
            addParameter(p, 'ShowProgressWindow', defaultShowProgressWindow);
            addParameter(p, 'L2WeightRegularization', defaultL2WeightRegularization);
            addParameter(p, 'SparsityRegularization', defaultSparsityRegularization);
            addParameter(p, 'SparsityProportion', defaultSparsityProportion);
            addParameter(p, 'ScaleData', defaultScaleData);
            addParameter(p, 'UseGPU', defaultUseGPU);
            
            parse(p, varargin{:});
            iAssertHiddenSizeIsScalarNumericRealGreaterThanZeroAndFinite(p.Results.HiddenSize);
            iAssertEncoderTransferFunctionIsOneOfTheseStrings( ...
                p.Results.EncoderTransferFunction, validEncoderTransferFunctions);
            iAssertDecoderTransferFunctionIsOneOfTheseStrings( ...
                p.Results.DecoderTransferFunction, validDecoderTransferFunctions);
            iAssertLossFunctionIsOneOfTheseStrings(p.Results.LossFunction, validLossFunctions);
            iAssertTrainingAlgorithmIsOneOfTheseStrings(p.Results.TrainingAlgorithm, validTrainingAlgorithms);
            iAssertMaxEpochsIsScalarNumericRealGreaterThanZeroAndFinite(p.Results.MaxEpochs);
            iAssertShowProgressWindowIsScalarAndReal(p.Results.ShowProgressWindow);
            iAssertL2WeightRegularizationIsValid(p.Results.L2WeightRegularization);
            iAssertSparsityRegularizationIsValid(p.Results.SparsityRegularization);
            iAssertSparsityProportionIsValid(p.Results.SparsityProportion);
            iAssertScaleDataIsScalarAndReal(p.Results.ScaleData);
            iAssertUseGPUIsScalarAndReal(p.Results.UseGPU);
            inputParametersStruct = p.Results;
        end
        
        function autoenc = train(X, net, useGPU)
            % train   Train the autoencoder
            %
            %   autonet = train(x, net, useGPU) takes a set of data x, a
            %   network object net, and a boolean value useGPU  and returns
            %   a trained autoencoder autonet.
            iAssertAutonet(net);
            iAssertThatDataIsNotGPUArray(X);
            [X,visualizationDimensions, trainedOnImages] = iConvertToDouble(X);
            %I modify this
            %parameter.----------------------------------------------------------------------------------
            gg = size(X,2);
            XX = X(:,gg/2+1:gg);
            XX = [XX,XX];
            net = train(net,X,XX,'useGPU',iYesOrNo(useGPU));
            autoenc = Autoencoder(net, useGPU, visualizationDimensions, trainedOnImages);
        end
        
        function net = createNetwork(paramsStruct)
            net = network;
            
            % Define topology
            net.numInputs = 1;
            net.numLayers = 2;
            net.biasConnect = [1; 1];
            net.inputConnect(1,1) = 1;
            net.layerConnect(2,1) = 1;
            net.outputConnect = [0 1];
            
            % Set values for labels
            net.name = 'Autoencoder';
            net.layers{1}.name = 'Encoder';
            net.layers{2}.name = 'Decoder';
            
            % Set up initialization options
            net.layers{1}.initFcn = 'initwb';
            net.inputWeights{1,1}.initFcn = 'randsmall';
            net.biases{1}.initFcn = 'randsmall';
            net.layers{2}.initFcn = 'initwb';
            net.layerWeights{2,1}.initFcn = 'randsmall';
            net.biases{2}.initFcn = 'randsmall';
            net.initFcn = 'initlay';
            
            % Set parameters
            net.layers{1}.size = paramsStruct.HiddenSize;
            net.layers{1}.transferFcn = paramsStruct.EncoderTransferFunction;
            net.layers{2}.transferFcn = paramsStruct.DecoderTransferFunction;
            net.divideFcn = 'dividetrain';
            net.performFcn = paramsStruct.LossFunction;
            net.trainFcn = paramsStruct.TrainingAlgorithm;
            net.trainParam.epochs = paramsStruct.MaxEpochs;
            if isdeployed
                % Do not show training GUI for deployed code
                net.trainParam.showWindow = false;
            else
                net.trainParam.showWindow = paramsStruct.ShowProgressWindow;
                % Add plot functions if code is not deployed
                net.plotFcns = {'plotperform'};
            end
            
            % Set up data scaling
            if(paramsStruct.ScaleData)
                getRange = str2func([paramsStruct.DecoderTransferFunction '.outputRange']);
                yrange = getRange();
                ymin = iIfInifiteChangeMagnitudeToOne(yrange(1));
                ymax = iIfInifiteChangeMagnitudeToOne(yrange(2));
                net.inputs{1}.processFcns = {'mapminmax'};
                net.inputs{1}.processParams{1}.ymax = ymax;
                net.inputs{1}.processParams{1}.ymin = ymin;
                net.outputs{end}.processFcns = {'mapminmax'};
                net.outputs{end}.processParams{1}.ymax = ymax;
                net.outputs{end}.processParams{1}.ymin = ymin;
            end
            
            % Set parameters specific to the sparse autoencoder
            net.performParam.L2WeightRegularization = paramsStruct.L2WeightRegularization;
            net.performParam.sparsityRegularization = paramsStruct.SparsityRegularization;
            net.performParam.sparsity = paramsStruct.SparsityProportion;
        end
    end
    
    methods(Access = private)
        function encoder = getEncoder(this)
            encoder = network;
            
            % Define topology
            encoder.numInputs = 1;
            encoder.numLayers = 1;
            encoder.inputConnect(1,1) = 1;
            encoder.outputConnect = 1;
            encoder.biasConnect = 1;
            
            % Set values for labels
            encoder.name = 'Encoder';
            encoder.layers{1}.name = 'Encoder';
            
            % Copy parameters from input network
            encoder.inputs{1}.size = this.Network.inputs{1}.size;
            encoder.layers{1}.size = this.HiddenSize;
            encoder.layers{1}.transferFcn = this.EncoderTransferFunction;
            encoder.IW{1,1} = this.EncoderWeights;
            encoder.b{1} = this.EncoderBiases;
            
            % Set a training function
            encoder.trainFcn = this.Network.trainFcn;
            
            % Set the input
            encoderStruct = struct(encoder);
            networkStruct = struct(this.Network);
            encoderStruct.inputs{1} = networkStruct.inputs{1};
            encoder = network(encoderStruct);
        end
        
        function decoder = getDecoder(this)
            decoder = network;
            
            % Define topology
            decoder.numInputs = 1;
            decoder.numLayers = 1;
            decoder.inputConnect(1,1) = 1;
            decoder.outputConnect = 1;
            decoder.biasConnect = 1;
            
            % Set values for labels
            decoder.name = 'Decoder';
            decoder.layers{1}.name = 'Decoder';
            
            % Copy parameters from input network
            decoder.inputs{1}.size = this.HiddenSize;
            decoder.layers{1}.size = this.Network.inputs{1}.size;
            decoder.layers{1}.transferFcn = this.DecoderTransferFunction;
            decoder.IW{1,1} = this.DecoderWeights;
            decoder.b{1} = this.DecoderBiases;
            
            % Set a training function
            decoder.trainFcn = this.Network.trainFcn;
            
            % Set the output
            decoderStruct = struct(decoder);
            networkStruct = struct(this.Network);
            decoderStruct.outputs{end} = networkStruct.outputs{end};
            decoder = network(decoderStruct);
        end
        
        function x = convertDataIntoAMatrixOfColumns(this, x)
            if(this.TrainedOnImages)
                if(iIsCellArrayOfEquallySizedImages(x))
                    if(iCellArrayOfImagesHasTheseDimensions(x, this.VisualizationDimensions))
                        x = iFormatCellArrayOfImagesIntoMatrix(x);
                    else
                        error(message('nnet:autoencoder:InputCellArrayDimensionsWrong'));
                    end
                elseif(iIsImageWithTheseDimensions(x, this.VisualizationDimensions))
                    x = x(:);
                else
                    error(message('nnet:autoencoder:InvalidInputDataForAutoencoderTrainedOnImages'));
                end
            else
                if(~iIsNumericAndRealMatrixWithThisNumberOfRows(x, size(this.EncoderWeights,2)))
                    error(message('nnet:autoencoder:InvalidInputDataForAutoencoderTrainedOnColumns'));
                end
            end
        end
        
        function y = convertMatrixOfColumnsIntoData(this, x, y)
            if(this.TrainedOnImages)
                if(iIsImageWithTheseDimensions(x, this.VisualizationDimensions))
                    y = reshape(y, this.VisualizationDimensions);
                else
                    y = iFormatMatrixOfImagesIntoCellArray(y, this.VisualizationDimensions);
                end
            else
            end
        end
    end
end

function result = iIsNumericAndReal(x)
result = isnumeric(x) && isreal(x);
end

function result = iIsScalarAndReal(x)
result = isscalar(x) && isreal(x);
end

function result = iHasThisNumberOfRows(x, numberOfRows)
result = (size(x,1) == numberOfRows);
end

function result = iIsNumericAndRealMatrixWithThisNumberOfRows(x, numberOfRows)
result = iIsNumericAndReal(x) && ismatrix(x) && iHasThisNumberOfRows(x, numberOfRows);
end

function result = iIsScalarNumericRealGreaterThanZeroAndFinite(x)
result = isscalar(x) && isnumeric(x) && isreal(x) && (x > 0) && isfinite(x);
end

function result = iIsScalarNumericRealGreaterThanOrEqualToZeroAndFinite(x)
result = isscalar(x) && isnumeric(x) && isreal(x) && (x >= 0) && isfinite(x);
end

function result = iIsScalarNumericRealAndBetweenZeroAndOneInclusive(x)
result = isscalar(x) && isnumeric(x) && isreal(x) && (x >= 0) && (x <= 1);
end

function result = iIsOneOfTheseStrings(x, strings)
result = any(strcmp(x, strings));
end

function iValidateCharacterArray(x)
validateattributes(x, {'char'}, {'row'})
end

function iValidateScalarLogical(x)
validateattributes(x, {'numeric','logical'}, {'scalar','binary'});
end

function iThrowErrorForInvalidMatrixOnlyValue(matrixOnly)
if(~matrixOnly)
    error(message('nnet:autoencoder:MatrixOnlyNotSupported'));
end
end

function iIssueWarningIfUserSetMatrixOnlyToTrue(parser)
if(userSetMatrixOnlyToTrue(parser))
    warning(message('nnet:autoencoder:MatrixOnlyToBeRemoved'));
end
end

function tf = userSetMatrixOnlyToTrue(parser)
tf = userSetMatrixOnly(parser) && parser.Results.MatrixOnly;
end

function tf = userSetMatrixOnly(parser)
tf = ~any(strcmp(parser.UsingDefaults, 'MatrixOnly'));
end

function iCallGenFunctionWithCorrectArguments(network, arguments)
genFunction(network, arguments.pathname, 'MatrixOnly', 'yes', 'ShowLinks', arguments.ShowLinks);
end

function exception = iCreateExceptionFromErrorID(errorID)
exception = MException(errorID, getString(message(errorID)));
end

function iAssertHiddenSizeIsScalarNumericRealGreaterThanZeroAndFinite(hiddenSize)
if iIsScalarNumericRealGreaterThanZeroAndFinite(hiddenSize)
else
    exception = iCreateExceptionFromErrorID('nnet:autoencoder:HiddenSizeIsInvalid');
    throwAsCaller(exception);
end
end

function iAssertEncoderTransferFunctionIsOneOfTheseStrings(encoderTransferFunction, validEncoderTransferFunctions)
if iIsOneOfTheseStrings(encoderTransferFunction, validEncoderTransferFunctions)
else
    exception = iCreateExceptionFromErrorID('nnet:autoencoder:EncoderTransferFunctionIsInvalid');
    throwAsCaller(exception);
end
end

function iAssertDecoderTransferFunctionIsOneOfTheseStrings(decoderTransferFunction, validDecoderTransferFunctions)
if iIsOneOfTheseStrings(decoderTransferFunction, validDecoderTransferFunctions)
else
    exception = iCreateExceptionFromErrorID('nnet:autoencoder:DecoderTransferFunctionIsInvalid');
    throwAsCaller(exception);
end
end

function iAssertLossFunctionIsOneOfTheseStrings(lossFunction, validLossFunctions)
if iIsOneOfTheseStrings(lossFunction, validLossFunctions)
else
    exception = iCreateExceptionFromErrorID('nnet:autoencoder:LossFunctionIsInvalid');
    throwAsCaller(exception);
end
end

function iAssertTrainingAlgorithmIsOneOfTheseStrings(trainingAlgorithm, validTrainingAlgorithms)
if iIsOneOfTheseStrings(trainingAlgorithm, validTrainingAlgorithms)
else
    exception = iCreateExceptionFromErrorID('nnet:autoencoder:TrainingAlgorithmIsInvalid');
    throwAsCaller(exception);
end
end

function iAssertMaxEpochsIsScalarNumericRealGreaterThanZeroAndFinite(maxEpochs)
if iIsScalarNumericRealGreaterThanZeroAndFinite(maxEpochs)
else
    exception = iCreateExceptionFromErrorID('nnet:autoencoder:MaxEpochsIsInvalid');
    throwAsCaller(exception);
end
end

function iAssertShowProgressWindowIsScalarAndReal(showProgressWindow)
if iIsScalarAndReal(showProgressWindow)
else
    exception = iCreateExceptionFromErrorID('nnet:autoencoder:ShowProgressWindowIsInvalid');
    throwAsCaller(exception);
end
end

function iAssertL2WeightRegularizationIsValid(L2WeightRegularization)
if iIsScalarNumericRealGreaterThanOrEqualToZeroAndFinite(L2WeightRegularization)
else
    exception = iCreateExceptionFromErrorID('nnet:autoencoder:L2WeightRegularizationIsInvalid');
    throwAsCaller(exception);
end
end

function iAssertSparsityRegularizationIsValid(sparsityRegularization)
if iIsScalarNumericRealGreaterThanOrEqualToZeroAndFinite(sparsityRegularization)
else
    exception = iCreateExceptionFromErrorID('nnet:autoencoder:SparsityRegularizationIsInvalid');
    throwAsCaller(exception);
end
end

function iAssertSparsityProportionIsValid(sparsityProportion)
if iIsScalarNumericRealAndBetweenZeroAndOneInclusive(sparsityProportion)
else
    exception = iCreateExceptionFromErrorID('nnet:autoencoder:SparsityProportionIsInvalid');
    throwAsCaller(exception);
end
end

function iAssertScaleDataIsScalarAndReal(scaleData)
if iIsScalarAndReal(scaleData)
else
    exception = iCreateExceptionFromErrorID('nnet:autoencoder:ScaleDataIsInvalid');
    throwAsCaller(exception);
end
end

function iAssertUseGPUIsScalarAndReal(useGPU)
if iIsScalarAndReal(useGPU)
else
    exception = iCreateExceptionFromErrorID('nnet:autoencoder:UseGPUIsInvalid');
    throwAsCaller(exception);
end
end

function result = iIsImage(image)
result = hasTwoOrThreeDimensions(image) && thirdDimensionIsOneOrThree(image);
end

function result = hasTwoOrThreeDimensions(x)
dimensions = size(x);
numberOfDimensions = numel(dimensions);
if((numberOfDimensions == 3) || (numberOfDimensions == 2))
    result = true;
else
    result = false;
end
end

function result = thirdDimensionIsOneOrThree(x)
sizeOfThirdDimension = size(x,3);
if((sizeOfThirdDimension == 3) || (sizeOfThirdDimension == 1))
    result = true;
else
    result = false;
end
end

function iAssertAutonet(autonet)
if ~(isequal(autonet.name, 'Autoencoder') && ...
        isequal(autonet.layers{1}.name, 'Encoder') && ...
        isequal(autonet.layers{2}.name, 'Decoder'))
    error(message('nnet:autoencoder:NetworkIsNotAnAutoencoder'));
end
end

function iAssertThatDataIsNotGPUArray(x)
if(isa(x,'gpuArray'))
    error(message('nnet:autoencoder:InputDataMustNotBeGPUArray'));
end
end

function result = iIsCellArrayOfEquallySizedImages(x)
if ~iscell(x)
    result = false;
    return
end
if isempty(x)
    result = false;
    return
end
if(~iIsImage(x{1}))
    result = false;
    return
end
imageDimensions = size(x{1});
if(~iCellArrayOfImagesHasTheseDimensions(x, imageDimensions))
    result = false;
    return
end
result = true;
end

function result = iCellArrayOfImagesHasTheseDimensions(x, imageDimensions)
result = true;
for i = 1:numel(x)
    if(~isequal(size(x{i}), imageDimensions))
        result = false;
    end
end
end

function result = iImageHasTheseDimensions(x, imageDimensions)
result = isequal(size(x), imageDimensions);
end

function result = iIsImageWithTheseDimensions(x, imageDimensions)
result = iIsImage(x) && iImageHasTheseDimensions(x, imageDimensions);
end

function [X,visualizationDimensions, trainedOnImages] = iConvertToDouble(X)
if(iIsNumericAndReal(X))
    visualizationDimensions = [1 size(X,1)];
    trainedOnImages = false;
elseif(iIsCellArrayOfEquallySizedImages(X))
    visualizationDimensions = size(X{1});
    trainedOnImages = true;
    X = iFormatCellArrayOfImagesIntoMatrix(X);
else
    error(message('nnet:autoencoder:InvalidInputData'));
end
end

function matrixOfImages = iFormatCellArrayOfImagesIntoMatrix(cellArrayOfImages)
sizeOfImage = numel(cellArrayOfImages{1});
numberOfImages = numel(cellArrayOfImages);
matrixOfImages = zeros(sizeOfImage, numberOfImages);
for i = 1:numberOfImages
    matrixOfImages(:,i) = cellArrayOfImages{i}(:);
end
end

function cellArrayOfImages = iFormatMatrixOfImagesIntoCellArray(matrixOfImages, imageDimensions)
numberOfImages =  size(matrixOfImages,2);
cellArrayOfImages = cell(1,numberOfImages);
for i = 1:numberOfImages
    cellArrayOfImages{i} = reshape(matrixOfImages(:,i), imageDimensions);
end
end

function result = iIsAutoencoder(autoenc)
result = isa(autoenc,'Autoencoder');
end

function result = iIsNetwork(net)
result = isa(net,'network');
end

function iAssertAutoencoderOrNetwork(net)
if ~iIsAutoencoder(net) && ~iIsNetwork(net)
    error(message('nnet:autoencoder:InputIsNotAnAutoencoderOrNetwork'));
end
end

function iAssertAllAutoencoderOrNetwork(varargin)
cellfun(@iAssertAutoencoderOrNetwork, varargin);
end

function y = iIfInifiteChangeMagnitudeToOne(x)
if x == Inf
    y = 1;
elseif x == -Inf
    y = -1;
else
    y = x;
end
end

function tf = iNetworkUsesMapMinMaxAtInput(net)
if(isempty(net.inputs{1}.processFcns))
    tf = false;
else
    tf = strcmp(net.inputs{1}.processFcns{1}, 'mapminmax');
end
end

function numImageChannels = iGetNumberOfImageChannels(visualizationDimensions)
if(numel(visualizationDimensions) == 3)
    numImageChannels = visualizationDimensions(3);
else
    numImageChannels = 1;
end
end

function [numVerticalImages, numHorizontalImages] = iCalculateGalleryDimensions(...
    imageHeight, imageWidth, numberOfWeightVectors)
numVerticalImages = ceil(sqrt(numberOfWeightVectors*(imageWidth+1)/(imageHeight+1)));
numHorizontalImages = ceil(numberOfWeightVectors/numVerticalImages);
end

function yesOrNo = iYesOrNo(tf)
if tf
    yesOrNo = 'yes';
else
    yesOrNo = 'no';
end
end