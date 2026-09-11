% dump_bdc_reg1.m
% Diagnostic harness for BDcreation_reg.m ("reg1", the RGB/decorrstretch +
% LAB k-means pipeline) parity work. Mirrors dump_bdc_reg2.m. Run with:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch \
%     "cd('.../tests/matlab_parity'); dump_bdc_reg1"
%
% Outputs under dumps/<case_id>/ :
%   images.mat          HEmoving (double) and fixedSHG as regmex sees it
%                       (double(uint8) -> 0..255)
%   intermediates.mat   every preprocessing intermediate
%   tform_similarity.txt, tform_affine.txt, meta.mat
%   registered.tif      the imwrite'd output; compared with the golden
%   kmeans_probe.txt    whether the collagen cluster depends on the RNG seed
%   optimization_trace.txt (captureTrace=true)

function dump_bdc_reg1(caseFilter, captureTrace, kmeansSeed, outSuffix)
    % dump_bdc_reg1()                         all cases, rng('default') before kmeans
    % dump_bdc_reg1('test8', true)            one case + DisplayOptimization trace
    % dump_bdc_reg1('test8', false, 3, '_s3') one case with rng(3) -> dumps/test8_s3
    if nargin < 1; caseFilter = ''; end
    if nargin < 2; captureTrace = false; end
    if nargin < 3; kmeansSeed = []; end
    if nargin < 4; outSuffix = ''; end
    thisDir = fileparts(mfilename('fullpath'));
    fixtureRoot = fullfile(thisDir, '..', 'test_for_shg_he_registration_BDcreation');
    p02 = fullfile(fixtureRoot, 'new_test_datasets_tests4-5-6-7');
    outRoot = fullfile(thisDir, 'dumps');
    if ~exist(outRoot, 'dir'); mkdir(outRoot); end

    cases = {
        % case_id, he_dir, he_file, shg_dir, ppm, golden_dir
        'test8', fullfile(p02, 'HE'), 'patient_02_roi4.tif', fullfile(p02, 'SHG'), 3.0, ...
            fullfile(p02, 'HE', 'HE_registered_for_reg1_test6b_ppm3');
        'test9', fullfile(p02, 'HE'), 'patient_02_roi4.tif', fullfile(p02, 'SHG'), 2.6, ...
            fullfile(p02, 'HE', 'HE_registered_for_reg1_test9_ppm2p6');
    };

    for i = 1:size(cases, 1)
        caseId = cases{i, 1};
        if ~isempty(caseFilter) && ~strcmp(caseId, caseFilter)
            continue;
        end
        fprintf('\n===== DUMP %s ppm=%.2f (reg1) =====\n', caseId, cases{i, 5});
        dump_one([caseId outSuffix], cases{i, 2}, cases{i, 3}, cases{i, 4}, cases{i, 5}, ...
            cases{i, 6}, outRoot, captureTrace, kmeansSeed);
    end
    fprintf('\nAll reg1 dumps written under %s\n', outRoot);
end


function dump_one(caseId, heDir, heFile, shgDir, pixelpermicron, goldenDir, outRoot, captureTrace, kmeansSeed)
    outDir = fullfile(outRoot, caseId);
    if ~exist(outDir, 'dir'); mkdir(outDir); end

    HE_image = fullfile(heDir, heFile);
    SHGdata = fullfile(shgDir, heFile);
    if ~exist(HE_image, 'file') || ~exist(SHGdata, 'file')
        warning('Missing inputs for %s; skip', caseId);
        return;
    end

    % ---- BDcreation_reg.m, verbatim apart from variable capture ----
    fixedSHG_raw = imread(SHGdata);
    fixedSHG = imadjust(fixedSHG_raw);

    HEdata = imread(HE_image);
    HEdata = im2double(HEdata);
    max_HEdata = max(max(max(HEdata)));
    HEdata_adj = imadjust(HEdata, [0 max_HEdata], [0 1]);
    HEdata_adj = im2uint8(HEdata_adj);
    HEdata_adj0 = HEdata_adj;              % pre-decorrstretch (uint8)
    HEdata_nuclei = HEdata_adj;
    HEdata_red = HEdata_adj;
    HEdata_gray = rgb2gray(HEdata_adj);
    HERGB = HEdata_adj;

    S = decorrstretch(HERGB, 'tol', 0.01);
    HEdata_adj = S;

    [m, n] = size(HEdata_gray);
    for i = 1:m
        for j = 1:n
            if (HEdata_adj(i,j,1) < 120 && HEdata_adj(i,j,2) > 150 && HEdata_adj(i,j,3) < 120)
                HEdata_nuclei(i,j,:) = HEdata_adj(i,j,:);
            else
                HEdata_nuclei(i,j,:) = 0;
            end
            if (HEdata_adj(i,j,1) > 200 && HEdata_adj(i,j,2) < 100 && HEdata_adj(i,j,3) > 100)
                HEdata_red(i,j,1) = HEdata_adj(i,j,1);
            else
                HEdata_red(i,j,:) = 0;
            end
        end
    end

    cform = makecform('srgb2lab');
    lab_HEdata = applycform(HEdata_red, cform);
    ab = double(lab_HEdata(:,:,2:3));
    nrows = size(ab, 1);
    ncols = size(ab, 2);
    ab = reshape(ab, nrows * ncols, 2);
    nColors = 3;

    % RNG probe: does the selected collagen cluster depend on the seed?
    % BDcreation_reg.m never seeds, so the golden was produced from whatever
    % state the session had. If every seed gives the same mask, the Python
    % port only needs *a* correct k-means, not MATLAB's RNG stream.
    probeFile = fullfile(outDir, 'kmeans_probe.txt');
    fid = fopen(probeFile, 'w');
    ref_mask = [];
    seeds = [0 1 2 3 42 12345];
    for s = seeds
        rng(s, 'twister');
        [ci, cc] = kmeans(ab, nColors, 'distance', 'sqEuclidean', 'Replicates', 3);
        [~, ix] = sort(mean(cc, 2));
        mask_s = reshape(ci, nrows, ncols) == ix(nColors);
        if isempty(ref_mask); ref_mask = mask_s; end
        fprintf(fid, 'seed=%d same_collagen_mask_as_seed0=%d n_collagen=%d centers=%s\n', ...
            s, isequal(mask_s, ref_mask), nnz(mask_s), mat2str(cc(ix, :), 6));
    end
    fclose(fid);

    % The real run: default session RNG state as in matlab -batch, unless a
    % seed was requested to reproduce an alternative k-means optimum.
    if isempty(kmeansSeed)
        rng('default');
    else
        rng(kmeansSeed, 'twister');
    end
    [cluster_idx, cluster_center] = kmeans(ab, nColors, 'distance', 'sqEuclidean', ...
        'Replicates', 3);
    pixel_labels = reshape(cluster_idx, nrows, ncols);
    segmented_images = cell(1, 3);
    rgb_label = repmat(pixel_labels, [1 1 3]);
    for k = 1:nColors
        color = HEdata;
        color(rgb_label ~= k) = 0;
        segmented_images{k} = color;
    end
    mean_cluster_value = mean(cluster_center, 2);
    [~, idx] = sort(mean_cluster_value);
    collagen_cluster = idx(nColors);
    HE_collagen = im2double(rgb2gray(segmented_images{idx(nColors)}));
    gray_nuclei = im2double(rgb2gray(HEdata_nuclei));
    h_nuclei = fspecial('gaussian', floor(pixelpermicron), 0.5);
    nuclei_filtered = imfilter(im2double(gray_nuclei), h_nuclei);
    BW_nuclei = im2bw(im2double(nuclei_filtered), 0.001); %#ok<IM2BW>
    BW_nuclei_discard = bwareaopen(BW_nuclei, ceil(50 * pixelpermicron^2));
    se = strel('disk', floor(pixelpermicron));
    BW_nuclei_dilated = imdilate(BW_nuclei_discard, se);
    BW_nuclei_filled = imfill(BW_nuclei_dilated, 'holes');
    HE_collagen_exclude0 = HE_collagen .* (~BW_nuclei_filled);
    HE_collagen_BW = im2bw(HE_collagen_exclude0, 0.01); %#ok<IM2BW>
    BW_discard = bwareaopen(HE_collagen_BW, ceil(pixelpermicron^2));
    HE_collagen_exclude = HE_collagen_exclude0 .* BW_discard;

    HEmoving = imresize(HE_collagen_exclude, size(fixedSHG));
    [optimizer, metric] = imregconfig('multimodal');
    optimizer.InitialRadius = optimizer.InitialRadius / 3.5;
    imregister(HEmoving, fixedSHG, 'affine', optimizer, metric); %#ok<NASGU>
    optimizer.MaximumIterations = 700;

    if captureTrace
        traceFile = fullfile(outDir, 'optimization_trace.txt');
        if exist(traceFile, 'file'); delete(traceFile); end
        diary(traceFile); diary on;
        fprintf('### STAGE similarity\n');
        tformSimilarity = imregtform(HEmoving, fixedSHG, 'similarity', optimizer, metric, ...
            'DisplayOptimization', true);
        fprintf('### STAGE affine\n');
        tform = imregtform(HEmoving, fixedSHG, 'affine', optimizer, metric, ...
            'InitialTransformation', tformSimilarity, 'DisplayOptimization', true);
        fprintf('### END\n');
        diary off;
    else
        tformSimilarity = imregtform(HEmoving, fixedSHG, 'similarity', optimizer, metric);
        tform = imregtform(HEmoving, fixedSHG, 'affine', optimizer, metric, ...
            'InitialTransformation', tformSimilarity);
    end
    RfixedSHG = imref2d(size(fixedSHG));
    HERmoving = imref2d(size(HEmoving));
    HEdata_registered = imresize(HEdata, size(fixedSHG));
    B = imwarp(HEdata_registered, HERmoving, tform, 'OutputView', RfixedSHG, 'FillValues', 255);
    registered_image = B;
    % ---- end BDcreation_reg.m ----

    imwrite(registered_image, fullfile(outDir, 'registered.tif'));
    fixedSHG_double = double(fixedSHG);   % exactly what imregtform hands to regmex
    save(fullfile(outDir, 'images.mat'), 'HEmoving', 'fixedSHG_double', '-v7');
    save(fullfile(outDir, 'intermediates.mat'), ...
        'fixedSHG_raw', 'fixedSHG', 'HEdata', 'max_HEdata', 'HEdata_adj0', 'S', ...
        'HEdata_nuclei', 'HEdata_red', 'lab_HEdata', 'pixel_labels', 'cluster_center', ...
        'collagen_cluster', 'HE_collagen', 'gray_nuclei', 'h_nuclei', 'nuclei_filtered', ...
        'BW_nuclei', 'BW_nuclei_discard', 'BW_nuclei_dilated', 'BW_nuclei_filled', ...
        'HE_collagen_exclude0', 'HE_collagen_BW', 'BW_discard', 'HE_collagen_exclude', ...
        'HEmoving', 'HEdata_registered', 'B', '-v7');
    writematrix(tformSimilarity.T, fullfile(outDir, 'tform_similarity.txt'), 'Delimiter', ' ');
    writematrix(tform.T, fullfile(outDir, 'tform_affine.txt'), 'Delimiter', ' ');

    meta = struct();
    meta.caseId = caseId;
    meta.pipeline = 'reg1';
    meta.kmeansSeed = kmeansSeed;
    meta.ppm_input = pixelpermicron;
    meta.HEmoving_size = size(HEmoving);
    meta.fixedSHG_size = size(fixedSHG);
    meta.fixedSHG_class = class(fixedSHG);
    meta.InitialRadius = optimizer.InitialRadius;
    meta.GrowthFactor = optimizer.GrowthFactor;
    meta.Epsilon = optimizer.Epsilon;
    meta.MaximumIterations = optimizer.MaximumIterations;
    meta.NumberOfHistogramBins = metric.NumberOfHistogramBins;
    meta.tformSimilarity_T = tformSimilarity.T;
    meta.tform_T = tform.T;
    meta.mask_coverage = mean(HEmoving(:) > 0);
    save(fullfile(outDir, 'meta.mat'), '-struct', 'meta');

    % Does this local run reproduce the committed golden?
    goldenFile = fullfile(goldenDir, heFile);
    if exist(goldenFile, 'file')
        G = imread(goldenFile);
        R = imread(fullfile(outDir, 'registered.tif'));
        if isequal(size(G), size(R))
            d = abs(double(G) - double(R));
            fprintf('  vs golden: MAE=%.4f exact=%.2f%% max=%d\n', ...
                mean(d(:)), 100 * mean(d(:) == 0), max(d(:)));
        else
            fprintf('  vs golden: SIZE MISMATCH %s vs %s\n', mat2str(size(G)), mat2str(size(R)));
        end
    end
    fprintf('  saved %s  mask_coverage=%.4f  sim.T(1,1)=%.6f  aff.T(1,1)=%.6f\n', ...
        outDir, meta.mask_coverage, tformSimilarity.T(1,1), tform.T(1,1));
end
