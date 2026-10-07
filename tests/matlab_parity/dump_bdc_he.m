% dump_bdc_he.m
% Instrumented BDcreationHE.m on MATLAB-registered HE (tests 1-3).
% kmeans is unseeded in the original; this dump pins rng(28,'twister')
% immediately before kmeans (same seed policy as dump_bdc_reg1.m).
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch ...
%     "cd('.../tests/matlab_parity'); dump_bdc_he"

function dump_bdc_he(caseFilter, kmeansSeed)
    if nargin < 1; caseFilter = ''; end
    if nargin < 2; kmeansSeed = 28; end
    thisDir = fileparts(mfilename('fullpath'));
    fixtureRoot = fullfile(thisDir, '..', 'test_for_shg_he_registration_BDcreation');
    outRoot = fullfile(thisDir, 'dumps');
    if ~exist(outRoot, 'dir'); mkdir(outRoot); end

    cases = {
        'test1', fullfile(fixtureRoot, 'HE', 'HE_registered_test1'), 'patient_001.tif', 1.5;
        'test2', fullfile(fixtureRoot, 'HE', 'HE_registered_test2'), 'patient_001.tif', 2.0;
        'test3', fullfile(fixtureRoot, 'HE', 'HE_registered_test3'), 'patient_001.tif', 3.0;
    };

    for i = 1:size(cases, 1)
        caseId = cases{i, 1};
        if ~isempty(caseFilter) && ~strcmp(caseId, caseFilter)
            continue;
        end
        dump_one_he(caseId, cases{i, 2}, cases{i, 3}, cases{i, 4}, outRoot, kmeansSeed);
    end
end


function dump_one_he(caseId, heDir, heFile, ppm, outRoot, kmeansSeed)
    outDir = fullfile(outRoot, ['he_' caseId]);
    if ~exist(outDir, 'dir'); mkdir(outDir); end
    IMGpath = fullfile(heDir, heFile);
    if ~exist(IMGpath, 'file')
        warning('Missing %s; skip %s', IMGpath, caseId);
        return;
    end
    fprintf('\n===== HE DUMP %s ppm=%.2f seed=%d =====\n', caseId, ppm, kmeansSeed);

    HEdata = imread(IMGpath);
    pixpermic = ppm;
    S = decorrstretch(HEdata, 'tol', 0.01);
    class_S = class(S);
    ab_gray = im2double(rgb2gray(HEdata));
    [m, n] = size(ab_gray);
    H = fspecial('disk', round(7 * pixpermic));
    class_k2 = '';
    class_k1 = '';
    % Do not preallocate k1/k3: MATLAB grows them from imfilter's class
    % (uint8 in -> uint8 out), which double zeros would change.
    for j = 1:3
        k = padarray(S(:,:,j), [70 70], 'symmetric');
        k2 = histeq(k);
        class_k2 = class(k2);
        k1(:,:,j) = imfilter(imfilter(k2, H), H); %#ok<AGROW>
        class_k1 = class(k1);
        k3(:,:,j) = k1(71:70+m, 71:70+n, j); %#ok<AGROW>
    end
    class_k3 = class(k3);
    ab = double(k3);
    nrows = size(ab, 1);
    ncols = size(ab, 2);
    ab_flat = reshape(ab, nrows * ncols, 3);
    nColors = 4;
    rng(kmeansSeed, 'twister');
    [cluster_idx, cluster_center] = kmeans(ab_flat, nColors, ...
        'distance', 'sqEuclidean', 'Replicates', 3);
    pixel_labels = reshape(cluster_idx, nrows, ncols);
    rgb_label = repmat(pixel_labels, [1 1 3]);
    mean_cluster_intensity = zeros(nColors, 1);
    segmented_images = cell(1, nColors);
    for k = 1:nColors
        color = k3;
        color(rgb_label ~= k) = 0;
        segmented_images{k} = color;
        mean_cluster_intensity(k, 1) = mean(nonzeros(rgb2gray(cell2mat(segmented_images(k)))));
    end
    mean_cluster_value = mean(cluster_center, 2);
    [~, idx] = sort(mean_cluster_value);
    [~, idx1] = sort(mean_cluster_intensity);
    cluster_val = zeros(nColors, 1);
    for k = 1:nColors
        cluster_val(k, 1) = find(idx == k) * find(idx1 == k);
    end
    [~, idx2] = sort(cluster_val);
    blue_cluster_num = idx2(1);

    epith_cell = im2double(cell2mat(segmented_images(blue_cluster_num)));
    epith_cell_BW = im2bw(rgb2gray(epith_cell), 0.001); %#ok<IM2BW>
    se = strel('disk', round(4 * pixpermic));
    epith_cell_BW_open = imdilate(epith_cell_BW, se);
    BWx = imfill(epith_cell_BW_open, 'holes');
    BWy = bwareaopen(~BWx, round((60 * pixpermic)^2));
    mask_image = bwareaopen(~BWy, round((35 * pixpermic)^2));
    BDmask = uint8(255 * mask_image);

    save(fullfile(outDir, 'images.mat'), ...
        'S', 'k3', 'pixel_labels', 'blue_cluster_num', 'cluster_center', ...
        'epith_cell_BW', 'BWx', 'mask_image', 'BDmask', ...
        'class_S', 'class_k2', 'class_k1', 'class_k3', 'kmeansSeed', '-v7');
    save(fullfile(outDir, 'intermediates.mat'), ...
        'H', 'ab_flat', 'mean_cluster_value', 'mean_cluster_intensity', ...
        'cluster_val', 'epith_cell', 'epith_cell_BW_open', 'BWy', '-v7');
    meta = struct();
    meta.caseId = caseId;
    meta.ppm_input = ppm;
    meta.kmeans_seed = kmeansSeed;
    meta.class_S = class_S;
    meta.class_k2 = class_k2;
    meta.class_k1 = class_k1;
    meta.class_k3 = class_k3;
    meta.BDmask_coverage = mean(BDmask(:) > 0);
    save(fullfile(outDir, 'meta.mat'), '-struct', 'meta');
    fprintf('  saved %s  class(S)=%s class(k2)=%s class(k1)=%s class(k3)=%s coverage=%.4f\n', ...
        outDir, class_S, class_k2, class_k1, class_k3, meta.BDmask_coverage);
end
