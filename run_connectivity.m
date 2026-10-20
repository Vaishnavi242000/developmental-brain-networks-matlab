
%% run_connectivity.m
% Developmental functional connectivity: children vs. adults
%
% Compares large-scale brain network organization in children and adults
% using movie-watching fMRI (Richardson et al., 2018) and the Schaefer
% 100-parcel atlas (7 Yeo networks).
%
% Pipeline: parcel time series -> nuisance regression -> band-pass filter
% (0.01-0.08 Hz) -> Fisher-z connectivity -> head-motion QC ->
% network-level statistics (Welch t-tests, FDR) -> graph metrics -> figures
%
% Requires MATLAB R2020a+ and the Image Processing Toolbox (niftiread).
% Run Datafile.py first to download the data into ./data.

clear; clc; close all;
%% 0. Settings
dataDir = 'data';
figDir  = fullfile(pwd, 'figures');
if ~isfolder(figDir), mkdir(figDir); end
band    = [0.01 0.08];   % Hz: low-frequency BOLD fluctuations
fdMax   = 0.5;           % mm: exclude subjects with mean framewise displacement above this
density = 0.15;          % keep the strongest 15% of edges for graph metrics

%% 1. Load manifest and atlas
M = readtable(fullfile(dataDir, 'manifest.csv'), 'Delimiter', ',', ...
    'ReadVariableNames', true, 'TextType', 'string', ...
    'VariableNamingRule', 'preserve');
L = readtable(fullfile(dataDir, 'atlas_labels.csv'), 'Delimiter', ',', ...
    'ReadVariableNames', true, 'TextType', 'string', ...
    'VariableNamingRule', 'preserve');
atlas = round(double(niftiread(fullfile(dataDir, 'atlas_schaefer100.nii'))));
K = height(L);
nSub = height(M);

% Network membership of each parcel, from labels like "7Networks_LH_Default_PFC_1"
netNames = ["Vis","SomMot","DorsAttn","SalVentAttn","Limbic","Cont","Default"];
nNet = numel(netNames);
netOfParcel = strings(K, 1);
for k = 1:K
    p = split(L.label(k), "_");
    netOfParcel(k) = p(3);
end
[~, netIdx] = ismember(netOfParcel, netNames);
assert(all(netIdx > 0), 'Unrecognised network name in atlas labels.');

%% 2. Per-subject processing
Z      = nan(K, K, nSub);    % Fisher-z connectivity matrices
meanFD = nan(nSub, 1);

for s = 1:nSub
    fprintf('Subject %2d/%d  %s\n', s, nSub, M.participant_id(s));

    % --- load the 4D fMRI image
    f = char(M.func(s));
    if endsWith(f, '.gz')
        tmp = gunzip(f, tempdir);  f = tmp{1};
    end
    V    = double(niftiread(f));
    info = niftiinfo(f);
    if exist('tmp', 'var'), delete(f); clear tmp; end
    TR = info.PixelDimensions(4);
    if TR <= 0 || TR > 10, TR = 2; end    % fallback if header lacks TR

    [nx, ny, nz, nt] = size(V);
    assert(isequal([nx ny nz], size(atlas)), 'Atlas and fMRI grids differ.');

    % --- average voxels within each parcel -> time x parcels
    ts = parcelMeans(reshape(V, [], nt), atlas(:), K);

    % --- confounds (motion, tissue signals, etc.) from fMRIPrep
    C = readtable(M.confounds(s), 'FileType', 'text', 'Delimiter', '\t', ...
                  'TreatAsMissing', 'n/a');
    names = C.Properties.VariableNames;
    if ismember('framewise_displacement', names)
        meanFD(s) = mean(C.framewise_displacement, 'omitnan');
        C.framewise_displacement = [];
    end
    X = table2array(C(:, vartype('numeric')));
    X(isnan(X)) = 0;
    assert(size(X, 1) == nt, 'Confound rows do not match number of volumes.');

    % --- clean the signal and compute connectivity
    ts = cleanSignal(ts, X, TR, band);
    R  = corrcoef(ts);
    R(1:K+1:end) = 0;
    Z(:, :, s) = atanh(R);
end

%% 3. Head-motion quality control
isChild = contains(lower(M.group), "child");
keep    = isnan(meanFD) | meanFD <= fdMax;
fprintf('\nMean FD: children %.3f mm, adults %.3f mm\n', ...
    mean(meanFD(isChild), 'omitnan'), mean(meanFD(~isChild), 'omitnan'));
fprintf('Excluded for motion (FD > %.2f mm): %d children, %d adults\n', ...
    fdMax, sum(~keep & isChild), sum(~keep & ~isChild));

ch = keep & isChild;
ad = keep & ~isChild;
fprintf('Analysed: %d children, %d adults\n\n', sum(ch), sum(ad));

%% 4. Network-level connectivity (within- and between-network means)
NetConn = nan(nNet, nNet, nSub);
for s = 1:nSub
    for a = 1:nNet
        for b = 1:nNet
            blk = Z(netIdx == a, netIdx == b, s);
            if a == b, blk = blk(triu(true(size(blk)), 1)); end
            NetConn(a, b, s) = mean(blk(:), 'omitnan');
        end
    end
end

% Adults vs children for each unique network pair, FDR-corrected
[ia, ib] = find(triu(true(nNet)));
nTests = numel(ia);
tVal = zeros(nTests, 1); pVal = zeros(nTests, 1); dVal = zeros(nTests, 1);
for i = 1:nTests
    x = squeeze(NetConn(ia(i), ib(i), ad));
    y = squeeze(NetConn(ia(i), ib(i), ch));
    [tVal(i), pVal(i), dVal(i)] = welchT(x, y);
end
qVal = fdrBH(pVal);
netStats = table(netNames(ia)', netNames(ib)', tVal, dVal, pVal, qVal, ...
    'VariableNames', {'NetworkA','NetworkB','t_adult_vs_child','cohens_d','p','q_FDR'});
netStats = sortrows(netStats, 'p');
disp('Network-level differences (adults - children), sorted by p:');
disp(head(netStats, 8));
writetable(netStats, 'network_stats.csv');

% Within-default-mode-network connectivity vs age
isDMN = netNames == "Default";
dmn = squeeze(NetConn(isDMN, isDMN, :));
[rAge, pAge] = pearson(M.age(keep), dmn(keep));
fprintf('Within-DMN connectivity vs age: r = %.2f, p = %.3g\n', rAge, pAge);

%% 5. Graph metrics on thresholded networks
gEff = nan(nSub, 1); clust = nan(nSub, 1);
for s = 1:nSub
    A = Z(:, :, s);
    good = ~all(isnan(A), 2);
    A = A(good, good);  A(isnan(A)) = 0;
    [gEff(s), clust(s)] = graphMetrics(A, density);
end
[tE, pE] = welchT(gEff(ad), gEff(ch));
[tC, pC] = welchT(clust(ad), clust(ch));
fprintf('Global efficiency (adult vs child): t = %.2f, p = %.3g\n', tE, pE);
fprintf('Mean clustering   (adult vs child): t = %.2f, p = %.3g\n', tC, pC);

subjTable = table(M.participant_id, M.age, M.group, meanFD, keep, dmn, gEff, clust, ...
    'VariableNames', {'participant_id','age','group','meanFD','included', ...
                      'withinDMN_z','globalEfficiency','clustering'});
writetable(subjTable, 'subject_metrics.csv');

%% 6. Figures
[~, order] = sort(netIdx);
counts = histcounts(netIdx, 0.5:1:nNet + 0.5);
bounds = cumsum(counts);
centers = bounds - counts / 2;

% Figure 1: group-average connectivity matrices
mAd = mean(Z(:, :, ad), 3, 'omitnan');
mCh = mean(Z(:, :, ch), 3, 'omitnan');
fig1 = figure('Position', [100 100 1500 450]);
tl = tiledlayout(1, 3, 'TileSpacing', 'compact');
mats   = {mAd, mCh, mAd - mCh};
titles = {sprintf('Adults (n = %d)', sum(ad)), ...
          sprintf('Children (n = %d)', sum(ch)), 'Adults - Children'};
lims   = {[-1 1], [-1 1], [-0.4 0.4]};
for i = 1:3
    nexttile;
    imagesc(mats{i}(order, order)); axis square; caxis(lims{i});
    colormap(gca, blueWhiteRed(256)); colorbar;
    hold on;
    for b = bounds(1:end-1)
        xline(b + 0.5, 'k-', 'LineWidth', 0.8); yline(b + 0.5, 'k-', 'LineWidth', 0.8);
    end
    set(gca, 'XTick', centers, 'XTickLabel', netNames, 'XTickLabelRotation', 45, ...
             'YTick', centers, 'YTickLabel', netNames, 'FontSize', 9);
    title(titles{i});
end
title(tl, 'Functional connectivity (Fisher z), Schaefer 100 parcels');
saveFig(fig1, fullfile(figDir, 'connectivity_matrices.png'));

% Figure 2: within-network connectivity by group
within = zeros(nSub, nNet);
for a = 1:nNet, within(:, a) = squeeze(NetConn(a, a, :)); end
mu = [mean(within(ad, :)); mean(within(ch, :))]';
se = [std(within(ad, :)) / sqrt(sum(ad)); std(within(ch, :)) / sqrt(sum(ch))]';
fig2 = figure('Position', [100 100 800 420]);
hb = bar(mu); hold on;
for g = 1:2
    errorbar(hb(g).XEndPoints, mu(:, g), se(:, g), 'k.', 'LineWidth', 1);
end
set(gca, 'XTickLabel', netNames); ylabel('Mean within-network z');
legend({'Adults', 'Children'}, 'Location', 'northwest'); box off;
title('Within-network connectivity (mean \pm SEM)');
saveFig(fig2, fullfile(figDir, 'within_network_by_group.png'));

% Figure 3: within-DMN connectivity vs age
fig3 = figure('Position', [100 100 560 420]);
co = colororder(fig3);      % match Figure 2: adults = color 1, children = color 2
scatter(M.age(ch), dmn(ch), 40, co(2,:), 'filled'); hold on;
scatter(M.age(ad), dmn(ad), 40, co(1,:), 'filled');
c = polyfit(M.age(keep), dmn(keep), 1);
xx = linspace(min(M.age(keep)), max(M.age(keep)), 100);
plot(xx, polyval(c, xx), 'k-', 'LineWidth', 1.5);
xlabel('Age (years)'); ylabel('Within-DMN connectivity (z)'); box off;
legend({'Children', 'Adults', 'Linear fit'}, 'Location', 'southeast');
title(sprintf('Default mode network vs age (r = %.2f, p = %.3g)', rAge, pAge));
saveFig(fig3, fullfile(figDir, 'dmn_vs_age.png'));

% Figure 4: graph metrics
grp = categorical(repmat("Children", nSub, 1));
grp(~isChild) = "Adults";
fig4 = figure('Position', [100 100 800 380]);
subplot(1, 2, 1); boxchart(grp(keep), gEff(keep)); ylabel('Global efficiency');
title(sprintf('p = %.3g', pE));
subplot(1, 2, 2); boxchart(grp(keep), clust(keep)); ylabel('Mean clustering coefficient');
title(sprintf('p = %.3g', pC));
sgtitle(sprintf('Graph metrics at %d%% edge density', round(100 * density)));
saveFig(fig4, fullfile(figDir, 'graph_metrics.png'));

fprintf('\nDone. Figures in %s, tables in network_stats.csv and subject_metrics.csv\n', figDir);

function ts = parcelMeans(Y, labels, K)
% Average voxel time series (voxels x time) within each atlas parcel.
    in = labels >= 1 & labels <= K;
    S  = sparse(labels(in), find(in), 1, K, numel(labels));
    n  = full(sum(S, 2));
    ts = (S * Y ./ n)';          % time x parcels
    ts(:, n == 0) = NaN;         % parcels with no voxels on this grid
end

function ts = cleanSignal(ts, X, TR, band)
% Regress out intercept, linear trend and confounds; band-pass; z-score.
    nt = size(ts, 1);
    D  = [ones(nt, 1), (1:nt)' / nt, zscoreCols(X)];
    ts = ts - D * (D \ ts);
    ts = bandpassFFT(ts, TR, band);
    ts = (ts - mean(ts)) ./ std(ts);
end

function Xz = zscoreCols(X)
% z-score columns, dropping constant columns.
    sd = std(X);
    ok = sd > 0;
    Xz = (X(:, ok) - mean(X(:, ok))) ./ sd(ok);
end

function y = bandpassFFT(x, TR, band)
% Zero-phase band-pass filter applied in the frequency domain.
    n = size(x, 1);
    f = (0:n-1)' / (n * TR);
    f = min(f, 1 / TR - f);      % fold negative-frequency bins onto [0, Nyquist]
    keepBins = f >= band(1) & f <= band(2);
    Xf = fft(x);
    Xf(~keepBins, :) = 0;
    y = real(ifft(Xf));
end

function [E, Cmean] = graphMetrics(W, density)
% Binary graph from the strongest edges; global efficiency and clustering.
    n = size(W, 1);
    W(1:n+1:end) = 0;
    vals = W(triu(true(n), 1));
    thr  = quantile_(vals, 1 - density);
    A = double(W >= thr);  A(1:n+1:end) = 0;
    G = graph(A);
    Dm = distances(G);
    invD = 1 ./ Dm;  invD(1:n+1:end) = 0;  invD(isinf(Dm)) = 0;
    E = sum(invD(:)) / (n * (n - 1));
    k = sum(A, 2);
    tri = diag(A^3);
    Ci = zeros(n, 1);
    ok = k > 1;
    Ci(ok) = tri(ok) ./ (k(ok) .* (k(ok) - 1));
    Cmean = mean(Ci);
end

function q = quantile_(v, p)
% Simple quantile (avoids the Statistics Toolbox).
    v = sort(v(~isnan(v)));
    q = v(max(1, round(p * numel(v))));
end

function [t, p, d] = welchT(x, y)
% Welch two-sample t-test (two-tailed) and Cohen's d.
    x = x(~isnan(x)); y = y(~isnan(y));
    nx = numel(x); ny = numel(y); vx = var(x); vy = var(y);
    t  = (mean(x) - mean(y)) / sqrt(vx / nx + vy / ny);
    df = (vx / nx + vy / ny)^2 / ((vx / nx)^2 / (nx - 1) + (vy / ny)^2 / (ny - 1));
    p  = betainc(df / (df + t^2), df / 2, 0.5);
    d  = (mean(x) - mean(y)) / sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2));
end

function [r, p] = pearson(x, y)
% Pearson correlation with two-tailed p-value.
    ok = ~isnan(x) & ~isnan(y);
    x = x(ok); y = y(ok); n = numel(x);
    c = corrcoef(x, y); r = c(1, 2);
    t = r * sqrt((n - 2) / (1 - r^2));
    p = betainc((n - 2) / (n - 2 + t^2), (n - 2) / 2, 0.5);
end

function q = fdrBH(p)
% Benjamini-Hochberg false discovery rate correction.
    p = p(:); m = numel(p);
    [ps, i] = sort(p);
    qs = ps .* m ./ (1:m)';
    qs = flipud(cummin(flipud(qs)));
    q = zeros(m, 1); q(i) = min(qs, 1);
end

function c = blueWhiteRed(n)
% Diverging colormap: blue (negative) - white (zero) - red (positive).
    h  = floor(n / 2);
    up = linspace(0, 1, h)';
    c  = [[up, up, ones(h, 1)]; [ones(h, 1), flipud(up), flipud(up)]];
end

function saveFig(fig, path)
    exportgraphics(fig, path, 'Resolution', 200);
end
