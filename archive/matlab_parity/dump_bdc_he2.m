% dump_bdc_he2.m
% Instrumented BDcreationHE2.m on MATLAB-registered HE (tests 1-3).
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch ...
%     "cd('.../tests/matlab_parity'); dump_bdc_he2"
%
% Per case dumps/he2_<case>/ :
%   images.mat  (tracked)  - arrays pytest compares
%   intermediates.mat (gitignored)
%   meta.mat (gitignored)

function dump_bdc_he2(caseFilter)
    if nargin < 1; caseFilter = ''; end
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
        dump_one_he2(caseId, cases{i, 2}, cases{i, 3}, cases{i, 4}, outRoot);
    end
end


function dump_one_he2(caseId, heDir, heFile, ppm, outRoot)
    outDir = fullfile(outRoot, ['he2_' caseId]);
    if ~exist(outDir, 'dir'); mkdir(outDir); end
    IMGpath = fullfile(heDir, heFile);
    if ~exist(IMGpath, 'file')
        warning('Missing %s; skip %s', IMGpath, caseId);
        return;
    end
    fprintf('\n===== HE2 DUMP %s ppm=%.2f =====\n', caseId, ppm);

    pixpermic = ppm;
    HE = im2double(imread(IMGpath));
    if (pixpermic > 2)
        HEdata = imresize(HE, 2 / pixpermic);
        pixpermic = 2;
    else
        HEdata = HE;
    end
    [orig_row, orig_col, ~] = size(HE);

    r = HEdata(:,:,1); g = HEdata(:,:,2); b = HEdata(:,:,3);
    mean_r = mean(mean(r)); mean_g = mean(mean(g)); mean_b = mean(mean(b));
    std_r = std(std(r)); std_g = std(std(g)); std_b = std(std(b));
    HIGH_IN_r = min(mean_r + 2 * std_r, 1);
    HIGH_IN_g = min(mean_g + 2 * std_g, 1);
    HIGH_IN_b = min(mean_b + 2 * std_b, 1);
    HEdata = imadjust(HEdata, [0 0 0; HIGH_IN_r HIGH_IN_g HIGH_IN_b], [0 0 0; 1 1 1]);
    he_adjusted = HEdata;

    HEgray = rgb2gray(HEdata);
    HEthresh = graythresh(HEgray);
    BW_gray = im2bw(HEgray, HEthresh); %#ok<IM2BW>

    I = rgb2hsv(HEdata);
    channel1Min = 0.500; channel1Max = 0.790;
    channel2Min = graythresh(I(:,:,2));
    BW = (I(:,:,1) >= channel1Min) & (I(:,:,1) <= channel1Max) & ...
         (I(:,:,2) >= channel2Min) & (I(:,:,2) <= 1.000);
    BW = bwareaopen(BW, 150);
    se = strel('disk', ceil(pixpermic / 2));
    BW_nuclei = imopen(BW, se);
    maskednucleiImage = HEdata;
    maskednucleiImage(repmat(~BW_nuclei, [1 1 3])) = 0;

    HEhsv = rgb2hsv(HEdata);
    channel1Min_c = 0.837; channel1Max_c = 0.066;
    channel2Min_c = graythresh(HEhsv(:,:,2));
    BW_collagen = ((HEhsv(:,:,1) >= channel1Min_c) | (HEhsv(:,:,1) <= channel1Max_c)) & ...
        (HEhsv(:,:,2) >= channel2Min_c) & (HEhsv(:,:,2) <= 1.000);
    BW_collagen = bwareaopen(BW_collagen, 100);
    se = strel('disk', ceil(pixpermic));
    BW_collagen = imdilate(BW_collagen, se);
    se = strel('disk', round(3 * pixpermic));
    BW_collagen1 = imclose(BW_collagen, se);
    BW_nobackground = (HEhsv(:,:,2) >= channel2Min_c);
    sat_thresh = channel2Min_c;

    epith_cell_BW = im2bw(rgb2gray(maskednucleiImage), 0.001) .* (~BW_collagen1) .* BW_nobackground; %#ok<IM2BW>
    se = strel('disk', round(5 * pixpermic));
    epith_cell_BW_open = imdilate(epith_cell_BW, se);
    BWx = imfill(epith_cell_BW_open, 'holes');
    BWy = bwareaopen(~BWx, round((60 * pixpermic)^2));
    mask_image = bwareaopen(~BWy, round((35 * pixpermic)^2));
    se = strel('disk', round(4 * pixpermic));
    mask_image1 = imdilate(mask_image, se) .* (~BW_collagen1);
    h = fspecial('gaussian', 101, 25);
    B = imfilter(mask_image1, h, 'replicate', 'corr');
    mask_temp = imresize(B, [orig_row, orig_col]);
    mask_thresh = graythresh(mask_temp);
    mask_bw = im2bw(mask_temp, mask_thresh); %#ok<IM2BW>
    BDmask = mask_bw;

    save(fullfile(outDir, 'images.mat'), ...
        'he_adjusted', 'BW_nuclei', 'maskednucleiImage', 'BW_collagen1', ...
        'BW_nobackground', 'sat_thresh', 'epith_cell_BW', 'BWx', ...
        'mask_image1', 'B', 'mask_temp', 'mask_thresh', 'BDmask', '-v7');
    save(fullfile(outDir, 'intermediates.mat'), ...
        'HE', 'HEdata', 'HIGH_IN_r', 'HIGH_IN_g', 'HIGH_IN_b', ...
        'HEthresh', 'BW_gray', 'BW', 'BW_collagen', 'epith_cell_BW_open', ...
        'BWy', 'mask_image', 'h', '-v7');
    meta = struct();
    meta.caseId = caseId;
    meta.ppm_input = ppm;
    meta.pixpermic_working = pixpermic;
    meta.orig_size = [orig_row, orig_col];
    meta.working_size = [size(HEdata, 1), size(HEdata, 2)];
    meta.BDmask_coverage = mean(BDmask(:) > 0);
    save(fullfile(outDir, 'meta.mat'), '-struct', 'meta');
    fprintf('  saved %s  coverage=%.4f  working=%dx%d  orig=%dx%d\n', ...
        outDir, meta.BDmask_coverage, meta.working_size(1), meta.working_size(2), ...
        orig_row, orig_col);
end
