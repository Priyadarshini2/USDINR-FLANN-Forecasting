%% run_all.m
%  DOLLAR EXCHANGE RATE PREDICTION USING FLANN - MATLAB implementation
%
%  This script mirrors the Python pipeline (src/run_project.py) for
%  environments where MATLAB is the required platform (as listed on the
%  resume). It reads the same real USD/INR dataset produced by the Python
%  data pipeline (data/usdinr_383.csv) so both implementations are
%  comparing apples to apples.
%
%  Requires: Statistics and Machine Learning Toolbox (fitrnet / mlp),
%            Econometrics Toolbox (arima)
%
%  Usage (from the matlab/ folder):
%      >> run_all
%
%  If data/usdinr_383.csv does not exist yet, run the Python data pipeline
%  first:
%      python ../src/download_data.py
%      python ../src/prepare_data.py

clear; clc; close all;

%% ---------------------------------------------------------------
%  1. Load data
%  ---------------------------------------------------------------
dataPath = fullfile('..', 'data', 'usdinr_383.csv');
if ~isfile(dataPath)
    error(['Dataset not found at %s.\n' ...
           'Run the Python data pipeline first:\n' ...
           '  python ../src/download_data.py\n' ...
           '  python ../src/prepare_data.py'], dataPath);
end

T = readtable(dataPath);
series = T.Close;

nLags  = 4;
nTrain = 335;
nTest  = 48;
expectedLen = nLags + nTrain + nTest;
assert(numel(series) == expectedLen, ...
    'Expected %d raw price points, found %d.', expectedLen, numel(series));

fprintf('Loaded %d raw USD/INR observations (%s to %s)\n', ...
    numel(series), datestr(T.Date(1)), datestr(T.Date(end)));

%% ---------------------------------------------------------------
%  2. Build lagged supervised frame
%  ---------------------------------------------------------------
nUsable = numel(series) - nLags;
X = zeros(nUsable, nLags);
y = zeros(nUsable, 1);
for i = 1:nUsable
    X(i, :) = series(i : i + nLags - 1);
    y(i)    = series(i + nLags);
end

Xtrain = X(1:nTrain, :);
ytrain = y(1:nTrain);
Xtest  = X(nTrain+1:end, :);
ytest  = y(nTrain+1:end);

% Scale to [-1, 1], fit on TRAIN ONLY
xMin = min(Xtrain(:)); xMax = max(Xtrain(:));
yMin = min(ytrain);    yMax = max(ytrain);

scaleX = @(A) 2 * (A - xMin) / (xMax - xMin) - 1;
unscaleY = @(a) (a + 1) / 2 * (yMax - yMin) + yMin;
scaleY = @(a) 2 * (a - yMin) / (yMax - yMin) - 1;

XtrainS = scaleX(Xtrain);
XtestS  = scaleX(Xtest);
ytrainS = scaleY(ytrain);

%% ---------------------------------------------------------------
%  3a. FLANN - functional (trigonometric) expansion + linear weights,
%      trained with gradient descent (mirrors src/flann_model.py exactly)
%  ---------------------------------------------------------------
nExpansions = 3;
learningRate = 0.12;
nEpochs = 4000;
l2 = 5e-5;

expand = @(A) functionalExpansion(A, nExpansions);

XtrainExp = expand(XtrainS);
XtestExp  = expand(XtestS);

nWeights = size(XtrainExp, 2);
rng(42);
w = (rand(nWeights, 1) - 0.5) * 0.2;
b = 0;
n = size(XtrainExp, 1);

for epoch = 1:nEpochs
    z = XtrainExp * w + b;
    err = z - ytrainS;
    gradW = (XtrainExp' * err) / n + l2 * w;
    gradB = mean(err);
    w = w - learningRate * gradW;
    b = b - learningRate * gradB;
end

flannPredS = XtestExp * w + b;
flannPred = unscaleY(flannPredS);

%% ---------------------------------------------------------------
%  3b. MLP - via fitrnet (Statistics and Machine Learning Toolbox)
%  ---------------------------------------------------------------
try
    mlpModel = fitrnet(XtrainS, ytrainS, ...
        "LayerSizes", [8 4], ...
        "Activations", "tanh", ...
        "Standardize", false);
    mlpPredS = predict(mlpModel, XtestS);
    mlpPred = unscaleY(mlpPredS);
catch ME
    warning('fitrnet unavailable (%s). Skipping MLP in MATLAB run.', ME.message);
    mlpPred = nan(size(ytest));
end

%% ---------------------------------------------------------------
%  3c. ARIMA(1,1,1) - walk-forward one-step-ahead (Econometrics Toolbox)
%  ---------------------------------------------------------------
try
    history = series(1 : nLags + nTrain);
    arimaPred = zeros(nTest, 1);
    for t = 1:nTest
        mdl = arima(1, 1, 1);
        fit = estimate(mdl, history, 'Display', 'off');
        arimaPred(t) = forecast(fit, 1, 'Y0', history);
        history = [history; series(nLags + nTrain + t)]; %#ok<AGROW>
    end
catch ME
    warning('Econometrics Toolbox unavailable (%s). Skipping ARIMA in MATLAB run.', ME.message);
    arimaPred = nan(size(ytest));
end

%% ---------------------------------------------------------------
%  4. Evaluate
%  ---------------------------------------------------------------
mape = @(yt, yp) mean(abs((yt - yp) ./ yt)) * 100;
rmse = @(yt, yp) sqrt(mean((yt - yp).^2));
mae  = @(yt, yp) mean(abs(yt - yp));

fprintf('\n===== MODEL COMPARISON (Test set, n=%d) =====\n', nTest);
fprintf('%-15s %10s %10s %10s\n', 'Model', 'MAPE(%)', 'RMSE', 'MAE');
fprintf('%-15s %10.4f %10.4f %10.4f\n', 'FLANN', mape(ytest, flannPred), rmse(ytest, flannPred), mae(ytest, flannPred));
if ~any(isnan(mlpPred))
    fprintf('%-15s %10.4f %10.4f %10.4f\n', 'MLP', mape(ytest, mlpPred), rmse(ytest, mlpPred), mae(ytest, mlpPred));
end
if ~any(isnan(arimaPred))
    fprintf('%-15s %10.4f %10.4f %10.4f\n', 'ARIMA(1,1,1)', mape(ytest, arimaPred), rmse(ytest, arimaPred), mae(ytest, arimaPred));
end

%% ---------------------------------------------------------------
%  5. Plot
%  ---------------------------------------------------------------
figure('Position', [100 100 900 500]);
plot(ytest, 'k-', 'LineWidth', 2, 'DisplayName', 'Actual'); hold on;
plot(flannPred, 'b--', 'DisplayName', 'FLANN');
if ~any(isnan(mlpPred))
    plot(mlpPred, 'r--', 'DisplayName', 'MLP');
end
if ~any(isnan(arimaPred))
    plot(arimaPred, 'g--', 'DisplayName', 'ARIMA(1,1,1)');
end
xlabel('Test day index'); ylabel('USD/INR');
title('USD/INR Forecast Comparison (MATLAB)');
legend('show', 'Location', 'best');
grid on;
saveas(gcf, fullfile('..', 'results', 'matlab_forecast_comparison.png'));

fprintf('\nSaved plot to results/matlab_forecast_comparison.png\n');

%% ---------------------------------------------------------------
%  Local function: trigonometric functional expansion (same basis as
%  the Python FLANN implementation)
%  ---------------------------------------------------------------
function Xexp = functionalExpansion(X, nExpansions)
    blocks = {X};
    for i = 1:nExpansions
        blocks{end+1} = sin(i * pi * X); %#ok<AGROW>
        blocks{end+1} = cos(i * pi * X); %#ok<AGROW>
    end
    Xexp = cat(2, blocks{:});
end
