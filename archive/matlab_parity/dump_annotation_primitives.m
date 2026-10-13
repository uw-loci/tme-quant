% dump_annotation_primitives.m
% Small MATLAB probes for BDcreationHE / HE2 primitive ports (histeq, disk,
% padarray, strel). Run:
%   /Applications/MATLAB_R2025b.app/bin/matlab -batch ...
%     "cd('.../tests/matlab_parity'); dump_annotation_primitives"
%
% Writes dumps/annotation_primitives.mat (tracked).

function dump_annotation_primitives()
    thisDir = fileparts(mfilename('fullpath'));
    outDir = fullfile(thisDir, 'dumps');
    if ~exist(outDir, 'dir'); mkdir(outDir); end

    disk_r1 = fspecial('disk', 1);
    disk_r2 = fspecial('disk', 2);
    disk_r3 = fspecial('disk', 3);
    disk_r4 = fspecial('disk', 4);
    disk_r5 = fspecial('disk', 5);
    disk_r7 = fspecial('disk', 7);
    disk_r11 = fspecial('disk', 11);
    disk_r14 = fspecial('disk', 14);
    disk_r21 = fspecial('disk', 21);
    disk_r10p5 = fspecial('disk', 10.5);

    strel_r1 = strel('disk', 1).Neighborhood;
    strel_r2 = strel('disk', 2).Neighborhood;
    strel_r3 = strel('disk', 3).Neighborhood;
    strel_r4 = strel('disk', 4).Neighborhood;
    strel_r5 = strel('disk', 5).Neighborhood;
    strel_r7 = strel('disk', 7).Neighborhood;
    strel_r11 = strel('disk', 11).Neighborhood;

    pad_src = uint8(reshape(1:16, [4 4])');
    pad_sym = padarray(pad_src, [2 2], 'symmetric');
    pad_dbl = padarray(im2double(pad_src), [2 2], 'symmetric');

    histeq_u8_in = uint8(mod((0:255)', 256));
    histeq_u8_out = histeq(histeq_u8_in);
    histeq_ramp = histeq(uint8(reshape(0:255, [16 16])));
    rng(1, 'twister');
    histeq_rand_in = uint8(randi([0 255], [32 32]));
    histeq_rand_out = histeq(histeq_rand_in);
    histeq_dbl_in = linspace(0, 1, 64)';
    histeq_dbl_out = histeq(histeq_dbl_in);

    im2bw_in = linspace(0, 1, 11);
    im2bw_out = im2bw(im2bw_in, 0.5); %#ok<IM2BW>

    save(fullfile(outDir, 'annotation_primitives.mat'), ...
        'disk_r1', 'disk_r2', 'disk_r3', 'disk_r4', 'disk_r5', ...
        'disk_r7', 'disk_r11', 'disk_r14', 'disk_r21', 'disk_r10p5', ...
        'strel_r1', 'strel_r2', 'strel_r3', 'strel_r4', 'strel_r5', ...
        'strel_r7', 'strel_r11', ...
        'pad_src', 'pad_sym', 'pad_dbl', ...
        'histeq_u8_in', 'histeq_u8_out', 'histeq_ramp', ...
        'histeq_rand_in', 'histeq_rand_out', ...
        'histeq_dbl_in', 'histeq_dbl_out', ...
        'im2bw_in', 'im2bw_out', '-v7');
    fprintf('wrote %s\n', fullfile(outDir, 'annotation_primitives.mat'));
end
