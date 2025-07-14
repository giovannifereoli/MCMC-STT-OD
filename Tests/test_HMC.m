%% HMC-BASED ORBIT DETERMINATION USING RANGE/RANGE-RATE MEASUREMENTS
clear; clc; close all;

%% Constants
mu_earth = 398600.4418; % km^3/s^2
r_canberra = [-4461.0; 2682.0; -3677.0]; % km (column vector)

%% True initial state [x, y, z, vx, vy, vz] in km and km/s
true_state = [7000; 0; 0; 0; 7.5; 1.0];
t_obs = linspace(0, 3600, 100); % 1 hour, 100 samples

Phi0_flat = reshape(eye(6), [], 1);
x_aug0 = [true_state; Phi0_flat]; % 6 + 36 = 42 states

%% Propagate true state
[~, y_out] = ode113(@(t, x) two_body(t, x, mu_earth), t_obs, x_aug0);

% Preallocate and vectorize measurements
r = y_out(:, 1:3)';
v = y_out(:, 4:6)';
dr = r - r_canberra;
rho = sqrt(sum(dr.^2, 1));
drho = sum(dr .* v, 1) ./ rho;
range_meas = rho' + 0.001 * randn(length(t_obs), 1);
rangerate_meas = drho' + 0.0001 * randn(length(t_obs), 1);

%% Define log-posterior
logpdf = @(x) logPosteriorOrbit(x, t_obs, range_meas, rangerate_meas, ...
    r_canberra, mu_earth);

%% Initial guess and sampler
startpoint = true_state + randn(6, 1);
smp = hmcSampler(logpdf, startpoint, 'NumSteps', 10, 'StepSize', 0.01, 'CheckGradient', true);

%% MAP estimation
[MAPpars, fitInfo] = estimateMAP(smp, 'VerbosityLevel', 2);

%% Tune sampler
[smp, tuneinfo] = tuneSampler(smp, 'Start', MAPpars, 'VerbosityLevel', 2);
accratio = tuneinfo.StepSizeTuningInfo.AcceptanceRatio;

%% Draw Samples
NumChains = 1; Burnin = 100; NumSamples = 1000000;
MAPpars = startpoint;
chains = cell(NumChains, 1);
parfor c = 1:NumChains
    chains{c} = drawSamples(smp, ...
        'Start', MAPpars + randn(size(MAPpars)), ...
        'Burnin', Burnin, ...
        'NumSamples', NumSamples, ...
        'VerbosityLevel', 2); 
end

%% Diagnostics
diags = diagnostics(smp, chains);

%% Plots
allSamples = vertcat(chains{:});

% Plot trace plots
figure;
for i = 1:6
    subplot(3,2,i)
    plot(allSamples(:,i));
    xlabel('Sample'); ylabel(sprintf('x_%d', i));
    title(sprintf('Trace plot for x_%d', i));
end

% Plot autocorrelation
figure;
for i = 1:6
    subplot(3,2,i)
    plot_autocorr(allSamples(:,i), 100);
    title(sprintf('Autocorr x_%d', i));
end

% Plot marginal histograms
figure;
for i = 1:6
    subplot(3,2,i)
    histogram(allSamples(:,i), 50);
    title(sprintf('Marginal x_%d', i));
end

% Corner-style scatter
figure;
plotmatrix(allSamples);
sgtitle('Posterior Scatter Matrix');

%% Dynamics Function
function dx = two_body(~, x, mu)
    r = x(1:3);
    v = x(4:6);
    Phi = reshape(x(7:end), 6, 6);

    a = -mu * r / norm(r)^3;
    A = [zeros(3), eye(3); ...
        -mu * (eye(3)/norm(r)^3 - 3 * (r * r') / norm(r)^5), zeros(3)];

    dPhi = A * Phi;
    dx = [v; a; dPhi(:)];
end

%% Log Posterior Function
function [logpdf, gradlogpdf] = logPosteriorOrbit(x0, t_obs, rho_meas, ...
    drho_meas, r_station, mu)

    % Propagate state and STM
    Phi0 = reshape(eye(6), [], 1);
    x_aug0 = [x0; Phi0];
    [~, y] = ode113(@(t, x) two_body(t, x, mu), t_obs, x_aug0);
    N = length(t_obs);

    % Residuals and Jacobians
    res = zeros(2*N, 1);
    dres_dx0 = zeros(2*N, 6);

    for i = 1:N
        r = y(i, 1:3)';
        v = y(i, 4:6)';
        Phi = reshape(y(i, 7:end), 6, 6);

        dr = r - r_station;
        rho = norm(dr);
        drho = dot(dr, v) / rho;

        % Avoid division by zero
        if rho < 1e-3
            rho = 1e-3;
        end

        % Standard deviations (keep consistent)
        sig_rho = 0.001;
        sig_drho = 0.0001;

        % Measurement model residuals: (observed - predicted)
        res(2*i-1) = (rho_meas(i) - rho) / sig_rho;
        res(2*i)   = (drho_meas(i) - drho) / sig_drho;

        % Jacobians of measurement w.r.t. state
        H_rho = [dr' / rho, zeros(1,3)];
        H_drho = [(v' / rho - drho * dr' / rho^2), dr' / rho];

        % Apply chain rule: d(residual) = -H * Phi
        dres_dx0(2*i-1,:) = -H_rho * Phi / sig_rho;
        dres_dx0(2*i,:)   = -H_drho * Phi / sig_drho;
    end

    % Log-likelihood
    loglik = -0.5 * (res' * res);
    dloglik = -dres_dx0' * res;

    % Prior
    prior_inv = diag(1e-8 * ones(6,1));
    logprior = -0.5 * (x0' * prior_inv * x0);
    dlogprior = -prior_inv * x0;

    % Posterior
    logpdf = loglik + logprior;
    gradlogpdf = dloglik + dlogprior;
end


% Plot autocorr
function plot_autocorr(x, max_lag)
    x = x - mean(x);
    n = length(x);
    acf = zeros(max_lag+1, 1);

    for lag = 0:max_lag
        acf(lag+1) = sum(x(1:end-lag) .* x(1+lag:end)) / (n - lag);
    end
    acf = acf / acf(1); % normalize

    stem(0:max_lag, acf, 'filled');
    xlabel('Lag'); ylabel('Autocorrelation');
end
