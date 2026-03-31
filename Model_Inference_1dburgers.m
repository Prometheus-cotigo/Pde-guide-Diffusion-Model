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
fprintf('Starting Inference for Target Nu = %.4f...\n', target_nu);

% Save directory
saveDir = 'inference_snapshots_burgers';
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
Dt = Bt / (2*dt);

dtPhys = 0.00350; % Physics guidance step size

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
    
    % B. Physcis-Guide Step
    u_phys_temp = u_guided_pre;
    u_real = u_phys_temp * scaleFactor;
    u_smooth = imgaussfilt(u_real, 0.9); 
    
    term_ut = Dt * u_smooth;
    term_ux = u_smooth * Dx';
    term_uxx = u_smooth * Dxx';
    
    % Burgers Residual: u_t + u*u_x - nu*u_xx
    phys_res = term_ut + (u_smooth .* term_ux) - (target_nu * term_uxx);
    res_norm = phys_res / scaleFactor;
    res_norm(1, :) = 0; % Ignore t=0
    
    u_phys_temp = u_phys_temp - dtPhys * res_norm;
    
    % C. BCs/ICs Enforcement
    u_phys_temp(:, 1) = 0;      
    u_phys_temp(:, end) = 0;    
    u_phys_temp(1, :) = u0_true_norm; 
    
    u_guided = u_phys_temp;
    
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