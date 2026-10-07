% dump_bdc_reg2.m
% Step-0 diagnostic harness: dump HEmoving, fixedSHG, transforms for
% BDcreation_reg2 parity work. Run with:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch \
%     "cd('.../tests/matlab_parity'); dump_bdc_reg2"
%
% Outputs under dumps/<case_id>/ :
%   HEmoving.tif, fixedSHG.tif, tform_similarity.txt, tform_affine.txt
%   meta.mat (optimizer params, pixpermic, shapes)
% Optionally runs case test2 twice for determinism check.

function dump_bdc_reg2(caseFilter, captureTrace)
    % dump_bdc_reg2()                    -> all cases + determinism re-run
    % dump_bdc_reg2('test1', true)       -> one case, with DisplayOptimization
    %                                       trace saved to optimization_trace.txt
    if nargin < 1; caseFilter = ''; end
    if nargin < 2; captureTrace = false; end
    thisDir = fileparts(mfilename('fullpath'));
    fixtureRoot = fullfile(thisDir, '..', 'test_for_shg_he_registration_BDcreation');
    outRoot = fullfile(thisDir, 'dumps');
    if ~exist(outRoot, 'dir'); mkdir(outRoot); end

    cases = {
        % case_id, he_dir, he_file, shg_dir, ppm
        'test1', fullfile(fixtureRoot, 'HE'), 'patient_001.tif', ...
            fullfile(fixtureRoot, 'SHG'), 1.5;
        'test2', fullfile(fixtureRoot, 'HE'), 'patient_001.tif', ...
            fullfile(fixtureRoot, 'SHG'), 2.0;
        'test3', fullfile(fixtureRoot, 'HE'), 'patient_001.tif', ...
            fullfile(fixtureRoot, 'SHG'), 3.0;
        'test4', fullfile(fixtureRoot, 'new_test_datasets_tests4-5-6-7', 'HE'), ...
            'patient_02_roi2.tif', ...
            fullfile(fixtureRoot, 'new_test_datasets_tests4-5-6-7', 'SHG'), 2.6;
        'test5', fullfile(fixtureRoot, 'new_test_datasets_tests4-5-6-7', 'HE'), ...
            'patient_02_roi4.tif', ...
            fullfile(fixtureRoot, 'new_test_datasets_tests4-5-6-7', 'SHG'), 1.5;
        'test6', fullfile(fixtureRoot, 'new_test_datasets_tests4-5-6-7', 'HE'), ...
            'patient_02_roi4.tif', ...
            fullfile(fixtureRoot, 'new_test_datasets_tests4-5-6-7', 'SHG'), 2.6;
        'test7', fullfile(fixtureRoot, 'new_test_datasets_tests4-5-6-7', 'HE'), ...
            'patient_02_roi5.tif', ...
            fullfile(fixtureRoot, 'new_test_datasets_tests4-5-6-7', 'SHG'), 2.6;
    };

    for i = 1:size(cases, 1)
        caseId = cases{i, 1};
        if ~isempty(caseFilter) && ~strcmp(caseId, caseFilter)
            continue;
        end
        heDir = cases{i, 2};
        heFile = cases{i, 3};
        shgDir = cases{i, 4};
        ppm = cases{i, 5};
        fprintf('\n===== DUMP %s ppm=%.2f =====\n', caseId, ppm);
        dump_one(caseId, heDir, heFile, shgDir, ppm, outRoot, captureTrace);
    end

    if isempty(caseFilter)
        % Determinism: run test2 a second time into dumps/test2_rerun/
        fprintf('\n===== DETERMINISM re-run test2 =====\n');
        dump_one('test2_rerun', cases{2, 2}, cases{2, 3}, cases{2, 4}, ...
            cases{2, 5}, outRoot, false);
    end

    fprintf('\nAll dumps written under %s\n', outRoot);
end


function dump_one(caseId, heDir, heFile, shgDir, ppm, outRoot, captureTrace)
    outDir = fullfile(outRoot, caseId);
    if ~exist(outDir, 'dir'); mkdir(outDir); end

    HE_img = fullfile(heDir, heFile);
    SHG_img = fullfile(shgDir, heFile);
    if ~exist(HE_img, 'file')
        warning('Missing HE %s; skip %s', HE_img, caseId);
        return;
    end
    if ~exist(SHG_img, 'file')
        warning('Missing SHG %s; skip %s', SHG_img, caseId);
        return;
    end

    pixpermic = ppm;
    HE = im2double(imread(HE_img));
    SHG = im2double(imread(SHG_img));
    if size(SHG, 3) > 1
        SHG = rgb2gray(SHG);
    end

    if (pixpermic > 2)
        fixedSHG = imresize(SHG, 2 / pixpermic);
        pixpermic = 2;
    else
        fixedSHG = SHG;
    end
    RGB = imresize(HE, size(fixedSHG));

    r = RGB(:,:,1); g = RGB(:,:,2); b = RGB(:,:,3);
    mean_r = mean(mean(r)); mean_g = mean(mean(g)); mean_b = mean(mean(b));
    std_r = std(std(r)); std_g = std(std(g)); std_b = std(std(b));
    HIGH_IN_r = min(mean_r + 2 * std_r, 1);
    HIGH_IN_g = min(mean_g + 2 * std_g, 1);
    HIGH_IN_b = min(mean_b + 2 * std_b, 1);
    HEdata = imadjust(RGB, [0 0 0; HIGH_IN_r HIGH_IN_g HIGH_IN_b], [0 0 0; 1 1 1]);

    % Nuclei (HSV)
    I = rgb2hsv(HEdata);
    channel2Min = graythresh(I(:,:,2));
    BW = (I(:,:,1) >= 0.500) & (I(:,:,1) <= 0.790) & ...
         (I(:,:,2) >= channel2Min) & (I(:,:,2) <= 1.000);
    BW = bwareaopen(BW, 150);
    se = strel('disk', ceil(pixpermic / 2));
    BW_nuclei = imopen(BW, se);
    maskednucleiImage = HEdata;
    maskednucleiImage(repmat(~BW_nuclei, [1 1 3])) = 0;

    % Collagen (HSV)
    HEhsv = rgb2hsv(HEdata);
    channel2Min_c = graythresh(HEhsv(:,:,2));
    BW_collagen = ((HEhsv(:,:,1) >= 0.837) | (HEhsv(:,:,1) <= 0.066)) & ...
        (HEhsv(:,:,2) >= channel2Min_c) & (HEhsv(:,:,2) <= 1.000);
    BW_collagen = bwareaopen(BW_collagen, 100);

    % Nuclei refine + collagen isolate
    gray_nuclei = im2double(rgb2gray(maskednucleiImage));
    h_nuclei = fspecial('gaussian', floor(pixpermic), 0.5);
    nuclei_filtered = imfilter(im2double(gray_nuclei), h_nuclei);
    BW_nuclei2 = im2bw(im2double(nuclei_filtered), 0.001); %#ok<IM2BW>
    BW_nuclei_discard = bwareaopen(BW_nuclei2, ceil(50 * pixpermic^2));
    se2 = strel('disk', floor(pixpermic));
    BW_nuclei_dilated = imdilate(BW_nuclei_discard, se2);
    BW_nuclei_filled = imfill(BW_nuclei_dilated, 'holes');
    HE_collagen_BW = BW_collagen .* (~BW_nuclei_filled);
    BW_discard = bwareaopen(HE_collagen_BW, ceil(pixpermic^2));
    HEmoving = HE_collagen_BW .* BW_discard;

    % Registration (BDcreation_reg2.m)
    [optimizer, metric] = imregconfig('multimodal');
    optimizer.InitialRadius = optimizer.InitialRadius / 3.5;
    % warm-up (discarded, as in original)
    imregister(HEmoving, fixedSHG, 'affine', optimizer, metric); %#ok<NASGU>
    optimizer.MaximumIterations = 700;

    if captureTrace
        % DisplayOptimization prints one line per ES iteration (all levels).
        traceFile = fullfile(outDir, 'optimization_trace.txt');
        if exist(traceFile, 'file'); delete(traceFile); end
        diary(traceFile);
        diary on;
        fprintf('### STAGE similarity\n');
        tformSimilarity = imregtform(HEmoving, fixedSHG, 'similarity', ...
            optimizer, metric, 'DisplayOptimization', true);
        fprintf('### STAGE affine\n');
        tform = imregtform(HEmoving, fixedSHG, 'affine', optimizer, metric, ...
            'InitialTransformation', tformSimilarity, 'DisplayOptimization', true);
        fprintf('### END\n');
        diary off;
    else
        tformSimilarity = imregtform(HEmoving, fixedSHG, 'similarity', ...
            optimizer, metric);
        tform = imregtform(HEmoving, fixedSHG, 'affine', optimizer, metric, ...
            'InitialTransformation', tformSimilarity);
    end

    % Save dumps. imwrite quantises doubles to uint8, so also keep the exact
    % double arrays the registration actually consumed.
    imwrite(HEmoving, fullfile(outDir, 'HEmoving.tif'));
    imwrite(fixedSHG, fullfile(outDir, 'fixedSHG.tif'));
    save(fullfile(outDir, 'images.mat'), 'HEmoving', 'fixedSHG', '-v7');
    % Every preprocessing intermediate, so a Python mask mismatch can be
    % attributed to a single step (imadjust / rgb2hsv / graythresh / strel...).
    save(fullfile(outDir, 'intermediates.mat'), ...
        'RGB', 'HIGH_IN_r', 'HIGH_IN_g', 'HIGH_IN_b', 'HEdata', ...
        'channel2Min', 'BW', 'BW_nuclei', 'channel2Min_c', 'BW_collagen', ...
        'gray_nuclei', 'h_nuclei', 'nuclei_filtered', 'BW_nuclei2', ...
        'BW_nuclei_discard', 'BW_nuclei_dilated', 'BW_nuclei_filled', ...
        'HE_collagen_BW', 'BW_discard', '-v7');
    writematrix(tformSimilarity.T, fullfile(outDir, 'tform_similarity.txt'), ...
        'Delimiter', ' ');
    writematrix(tform.T, fullfile(outDir, 'tform_affine.txt'), 'Delimiter', ' ');

    meta = struct();
    meta.caseId = caseId;
    meta.ppm_input = ppm;
    meta.pixpermic_working = pixpermic;
    meta.HEmoving_size = size(HEmoving);
    meta.fixedSHG_size = size(fixedSHG);
    meta.InitialRadius = optimizer.InitialRadius;
    meta.GrowthFactor = optimizer.GrowthFactor;
    meta.Epsilon = optimizer.Epsilon;
    meta.MaximumIterations = optimizer.MaximumIterations;
    meta.NumberOfHistogramBins = metric.NumberOfHistogramBins;
    meta.UseAllPixels = metric.UseAllPixels;
    meta.tformSimilarity_T = tformSimilarity.T;
    meta.tform_T = tform.T;
    meta.mask_coverage = mean(HEmoving(:) > 0);
    save(fullfile(outDir, 'meta.mat'), '-struct', 'meta');

    fprintf('  saved %s  mask_coverage=%.4f  sim.T(1,1)=%.6f  aff.T(1,1)=%.6f\n', ...
        outDir, meta.mask_coverage, tformSimilarity.T(1,1), tform.T(1,1));
end
