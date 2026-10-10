%% ============================================================
%  Step 3: PDE-Guided Inference (1D Burgers)
%  Function: Load Model, Perform Inference & Visualize
% =============================================================
clc; clear; close all;
rng(1);

%% ------------------------------------------------------------
% [Part 0] Load Model
% ------------------------------------------------------------
modelFileName = 'Burgers_Model.mat';
if ~exist(modelFileName, 'file')
    error('Please run DM_ModelTraining_1dburgers first!');
end
fprintf('Loading model...\n');
load(modelFileName); 

%% ------------------------------------------------------------
% [Part 1] Setup Inference
% ------------------------------------------------------------
target_nu = 0.0075;
guidanceMode = 'stable_energy'; % 'stable_energy' | 'legacy_residual' 
fprintf('Starting Inference for Target Nu = %.4f...\n', target_nu);

% Save directory
saveDir = ['inference_snapshots_burgers_' guidanceMode];
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

% Ground Truth IC for Constraint
u0_true = sin(2*pi*x_vec); 
u0_true_norm = u0_true / scaleFactor;

% Mesh and Finite Difference Operators
[X_mesh, T_mesh] = meshgrid(x_vec, t_vec);
clim_fixed = [-1, 1]; 

Bx = spdiags([-ones(Nx,1) zeros(Nx,1) ones(Nx,1)], -1:1, Nx, Nx);
Dx = Bx / (2*dx);
Lxx = spdiags([ones(Nx,1) -2*ones(Nx,1) ones(Nx,1)], -1:1, Nx, Nx);
Dxx = Lxx / (dx^2);
Bt = spdiags([-ones(Nt,1) zeros(Nt,1) ones(Nt,1)], -1:1, Nt, Nt);
Bt(1, 1:2) = [-1, 1]; Bt(end, end-1:end) = [-1, 1];
DtLegacy = Bt / (2*dt); % Retain the original one-sided stencil for ablation.
Bt(1, 1:2) = [-2, 2]; Bt(end, end-1:end) = [-2, 2];
Dt = Bt / (2*dt); % Correct endpoint derivative for the energy-based mode.

dtPhys = 0.00350; % Initial trial update size in normalized solution units.
maxBacktracks = 12;
armijoFactor = 1e-4;
gradientClipRms = 5;
if Nx < 3 || Nt < 3 || ~isfinite(dx) || dx <= 0 || ~isfinite(dt) || dt <= 0 ...
        || ~isfinite(scaleFactor) || scaleFactor == 0 || ~isfinite(target_nu) || target_nu <= 0
    error('Invalid spatial/time grid, normalization scale or viscosity.');
end
if ~ismember(guidanceMode, {'stable_energy', 'legacy_residual'})
    error('Unsupported guidanceMode: %s', guidanceMode);
end
validMask = false(Nt,Nx);
validMask(2:end,2:end-1) = true; % Interior x, time rows t > 0.
guidanceTrace = nan(numT,5); % [PDE RMS before, after, accepted step, accepted, trials]

%% ------------------------------------------------------------
% [Part 2] Reverse Diffusion Loop
% ------------------------------------------------------------
% Initial noise
u_start_noise = randn(Nt, Nx); 
u_guided = u_start_noise; 
u_unguided = u_start_noise; 

hFig = figure('Position',[50 100 1600 500], 'Name', 'Live Inference (Surf)');

for t = numT:-1:1
    
    % Time Embeddings
    t_norm = (t / numT) * 2 * pi;
    plane_sin = ones(Nt, Nx, 'single') * sin(t_norm);
    plane_cos = ones(Nt, Nx, 'single') * cos(t_norm);
    
    % Random noise 
    if t > 1; z = randn(Nt,Nx); else; z = 0; end
    
    % --- Path 1: Unguided ---
    input_un = cat(3, single(u_unguided), plane_sin, plane_cos);
    eps_pred_un = predict(net, reshape(input_un, Nt, Nx, 3, 1));
    eps_pred_un = double(squeeze(eps_pred_un));
    u_unguided = (u_unguided - beta(t)/sqrt(1-alphaBar(t))*eps_pred_un) / sqrt(alpha(t)) + sqrt(beta(t))*z;
    
    % --- Path 2: Guided ---
    % A. Denoise Step
    input_gd = cat(3, single(u_guided), plane_sin, plane_cos);
    eps_pred_gd = predict(net, reshape(input_gd, Nt, Nx, 3, 1));
    eps_pred_gd = double(squeeze(eps_pred_gd));
    u_guided_pre = (u_guided - beta(t)/sqrt(1-alphaBar(t))*eps_pred_gd) / sqrt(alpha(t)) + sqrt(beta(t))*z;
    
    % B. Stable physics correction (or exact legacy correction).
    if ~all(isfinite(u_guided_pre(:)))
        error('Non-finite guided denoising field at reverse step %d.', t);
    end
    if strcmp(guidanceMode, 'stable_energy')
        u_projected = enforceBurgersConstraints(u_guided_pre, u0_true_norm);
        [u_guided, energyBefore, energyAfter, acceptedStep, accepted, trials] = ...
            stableBurgersCorrection(u_projected, Dt, Dx, Dxx, target_nu, ...
                scaleFactor, validMask, dtPhys, maxBacktracks, ...
                armijoFactor, gradientClipRms);
        guidanceTrace(t,:) = [sqrt(2*energyBefore), sqrt(2*energyAfter), ...
            acceptedStep, double(accepted), trials];
        if mod(t,50) == 0 || t == 1
            fprintf('  Physics t=%d | PDE RMS %.4g -> %.4g | step %.3g | accepted %d/%d\n', ...
                t, guidanceTrace(t,1), guidanceTrace(t,2), acceptedStep, accepted, trials);
        end
    else
        % Preserve original residual-subtraction update, smoothing, and stencil.
        u_phys_temp = u_guided_pre;
        u_real = u_phys_temp * scaleFactor;
        u_smooth = imgaussfilt(u_real, 0.9);
        term_ut = DtLegacy * u_smooth;
        term_ux = u_smooth * Dx';
        term_uxx = u_smooth * Dxx';
        phys_res = term_ut + (u_smooth .* term_ux) - (target_nu * term_uxx);
        res_norm = phys_res / scaleFactor;
        res_norm(1,:) = 0;
        u_phys_temp = u_phys_temp - dtPhys * res_norm;
        u_phys_temp(:,1) = 0;
        u_phys_temp(:,end) = 0;
        u_phys_temp(1,:) = u0_true_norm;
        u_guided = u_phys_temp;
    end
    if ~all(isfinite(u_guided(:))) || ~all(isfinite(u_unguided(:)))
        error('Non-finite generated field at reverse step %d.', t);
    end

    % --- Save Snapshots ---
    if mod(t, 50) == 0 || t == 1
        hTemp = figure('Visible', 'off', 'Position', [0, 0, 512, 512], 'Color', 'w');
        surf(X_mesh, T_mesh, u_guided * scaleFactor);
        shading interp; view(2); axis tight; axis square; axis off;    
        clim(clim_fixed);    
        filename_gd = fullfile(saveDir, sprintf('guided_step_%04d.png', t));
        exportgraphics(hTemp, filename_gd, 'Resolution', 150); 
        close(hTemp); 
        fprintf('  Snapshot saved for t=%d.\n', t);
    end
    
    % --- Live Visualization ---
    if mod(t, 50) == 0 || t==1
        set(0, 'CurrentFigure', hFig);
        subplot(1,2,1); 
        surf(X_mesh, T_mesh, u_unguided * scaleFactor); 
        shading interp; view(2); axis tight; title('Unguided'); 
        colorbar; clim(clim_fixed); xlabel('Space x'); ylabel('Time t');
       
        subplot(1,2,2); 
        surf(X_mesh, T_mesh, u_guided * scaleFactor); 
        shading interp; view(2); axis tight; title(sprintf('Guided (Target \\nu=%.4f)', target_nu)); 
        colorbar; clim(clim_fixed); xlabel('Space x'); ylabel('Time t');
        drawnow;
    end
end
if strcmp(guidanceMode, 'stable_energy')
    fprintf('\nStable guidance: %d/%d corrections accepted, %d skipped.\n', ...
        sum(guidanceTrace(:,4)), numT, numT-sum(guidanceTrace(:,4)));
    fprintf('PDE RMS here excludes the initial-time row and spatial endpoints.\n');
end
u_final_guided = u_guided * scaleFactor;
u_final_unguided = u_unguided * scaleFactor;

%% ------------------------------------------------------------
% [Part 3] Validation & Analysis (2x3 Plot)
% ------------------------------------------------------------
fprintf('Solving Ground Truth for Target Nu = %.4f...\n', target_nu);

% Redefine local functions handles here (cannot call Step 1 directly)
m = 0;
icfun = @burgersIC_fixed;
bcfun = @burgersBC_fixed;
pdefun_gt = @(x,t,u,DuDx) burgersPDE(x,t,u,DuDx, target_nu);
sol_gt = pdepe(m, pdefun_gt, icfun, bcfun, x_vec, t_vec);
u_true = sol_gt(:,:,1);

abs_err_guided = abs(u_final_guided - u_true);
abs_err_unguided = abs(u_final_unguided - u_true);
rel_err_guided = norm(u_final_guided(:) - u_true(:)) / norm(u_true(:));
rel_err_unguided = norm(u_final_unguided(:) - u_true(:)) / norm(u_true(:));

fprintf('\nFinal Metrics:\n');
fprintf('  > Unguided Rel Error: %.4f\n', rel_err_unguided);
fprintf('  > Guided   Rel Error: %.4f\n', rel_err_guided);
if strcmp(guidanceMode, 'stable_energy')
    save(fullfile(saveDir, sprintf('stability_trace_nu_%.4f.mat',target_nu)), ...
        'guidanceTrace','target_nu','guidanceMode','dtPhys', ...
        'rel_err_guided','rel_err_unguided');
end

figure('Name', 'Core Analysis: Burgers Final', 'Position', [50, 50, 1600, 900]);
clim_data = [min(u_true(:)), max(u_true(:))];

% Row 1: Results
subplot(2,3,1); 
surf(X_mesh, T_mesh, u_final_unguided); 
shading interp; view(2); axis tight;
colorbar; clim(clim_data); 
colormap(gca, parula); 
title({'[Unguided] Result', sprintf('Rel Err: %.4f', rel_err_unguided)}); ylabel('Time t');

subplot(2,3,2); 
surf(X_mesh, T_mesh, u_final_guided); 
shading interp; view(2); axis tight;
colorbar; clim(clim_data); 
colormap(gca, parula); 
title({'[Guided] Result', sprintf('Rel Err: %.4f', rel_err_guided)});

subplot(2,3,3); 
surf(X_mesh, T_mesh, u_true); 
shading interp; view(2); axis tight;
colorbar; clim(clim_data); 
colormap(gca, parula); 
title({'[Ground Truth]', sprintf('Target \\nu=%.4f', target_nu)});

% Row 2: Errors and Cross-section
subplot(2,3,4); 
surf(X_mesh, T_mesh, abs_err_unguided); 
shading interp; view(2); axis tight;
colormap(gca, jet); 
colorbar; 
title({'[Unguided] Error', '(Jet / Auto Scale)'}); ylabel('Time t'); xlabel('Space x');

subplot(2,3,5); 
surf(X_mesh, T_mesh, abs_err_guided); 
shading interp; view(2); axis tight;
colormap(gca, jet); 
colorbar; 
title({'[Guided] Error', '(Jet / Auto Scale)'}); xlabel('Space x');

subplot(2,3,6);
plot(x_vec, u_final_unguided(end,:), 'b--', 'LineWidth', 1.5); hold on;
plot(x_vec, u_final_guided(end,:), 'r-', 'LineWidth', 2); 
plot(x_vec, u_true(end,:), 'k:', 'LineWidth', 2);
legend('Unguided', 'Guided', 'Truth', 'Location', 'best');
title('Cross-section at Final Time'); grid on; xlim([0,1]); xlabel('Space x');

sgtitle(sprintf('Core Analysis: Burgers Eq (Target \\nu=%.4f)', target_nu), 'FontSize', 16, 'FontWeight', 'bold');

%% ============================================================
%  Local Functions for Inference
% ============================================================
function [c,f,s] = burgersPDE(x,t,u,DuDx, nu)
    c = 1;
    f = nu * DuDx;
    s = -u * DuDx;
end
function u0 = burgersIC_fixed(x)
    u0 = sin(2*pi*x); 
end
function [pl,ql,pr,qr] = burgersBC_fixed(xl,ul,xr,ur,t)
    pl = ul; ql = 0;
    pr = ur; qr = 0;
end
% -------------------------------------------------------------------------
% Experimental stable correction: reduce discrete UNSMOOTHED PDE residual.
% This DOES NOT guarantee better accuracy against the pdepe reference.
% -------------------------------------------------------------------------
function [uNew, E0, Enew, acceptedStep, accepted, trials] = ...
    stableBurgersCorrection(u, Dt, Dx, Dxx, nu, scaleFactor, ...
        validMask, initialStep, maxBacktracks, armijoFactor, gradientClipRms)
    [E0, grad] = burgersEnergyAndGradient(u, Dt, Dx, Dxx, nu, scaleFactor, validMask);
    if ~isfinite(E0) || ~all(isfinite(grad(:)))
        error('Non-finite PDE energy or gradient before correction.');
    end
    uNew = u; Enew = E0; acceptedStep = 0; accepted = false; trials = 0;
    gradRms = sqrt(mean(grad(validMask).^2));
    if E0 <= eps || gradRms <= 1e-12
        return;
    end
    % Rescale the steepest-descent direction and clip extreme entries.
    direction = grad / gradRms;
    direction = max(-gradientClipRms, min(gradientClipRms, direction));
    directionalDerivative = sum(grad(:).*direction(:));
    if ~isfinite(directionalDerivative) || directionalDerivative <= 0
        return;
    end
    step = initialStep;
    for j = 1:maxBacktracks
        trials = j;
        candidate = u - step*direction;
        trialEnergy = burgersEnergyAndGradient(candidate, Dt, Dx, Dxx, ...
            nu, scaleFactor, validMask);
        if isfinite(trialEnergy) && ...
                trialEnergy <= E0 - armijoFactor*step*directionalDerivative
            uNew = candidate;
            Enew = trialEnergy;
            acceptedStep = step;
            accepted = true;
            return;
        end
        step = 0.5*step;
    end
    % If every trial fails, retain the constraint-projected denoiser output.
end

function [E, grad] = burgersEnergyAndGradient(u, Dt, Dx, Dxx, nu, scaleFactor, validMask)
    ux = u*Dx';
    % For u_real = scaleFactor*u, r = R(u_real)/scaleFactor.
    r = Dt*u + scaleFactor*(u.*ux) - nu*(u*Dxx');
    r(~validMask) = 0;
    N = nnz(validMask);
    E = 0.5*sum(r(:).^2)/N;
    if nargout > 1
        % Exact discrete Jacobian-transpose residual (w.r.t. normalized u).
        grad = (Dt'*r + scaleFactor*(r.*ux + (r.*u)*Dx) - nu*(r*Dxx))/N;
        grad(~validMask) = 0; % No updates to constrained initial/boundary nodes.
    end
end

function u = enforceBurgersConstraints(u, u0_true_norm)
    u(:,1) = 0;
    u(:,end) = 0;
    u(1,:) = u0_true_norm;
end
