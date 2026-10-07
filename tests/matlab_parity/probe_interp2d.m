function probe_interp2d()
% Probe the edge rule of MATLAB's compiled bilinear resampler used by imwarp
% (images.internal.interp2d -> imagesbuiltinImageInterpolation2D), so the
% Python matlab_imwarp_bilinear can match it at the outermost pixel band.
%
% Output: dumps/interp2d_probe.mat with img (4x5 double), X, Y (1-based
% intrinsic sample coords, dense grid from -1 to 7 step 0.05), out_linear.
here = fileparts(mfilename('fullpath'));
outDir = fullfile(here, 'dumps');
if ~exist(outDir, 'dir'); mkdir(outDir); end

img = reshape(1:20, 5, 4)' / 20;          % 4 rows x 5 cols, values in (0,1]
[X, Y] = meshgrid(-1:0.05:7, -1:0.05:6);   % 1-based intrinsic coordinates
fill = 255;
out_linear = images.internal.interp2d(img, X, Y, 'linear', fill);

% Also run a real imwarp with an identity+shift transform so the whole
% imwarp -> interp2d chain (including how imwarp builds X/Y) is covered.
tform = affine2d([1 0 0; 0 1 0; 0.3 -0.2 1]);
Rin = imref2d(size(img));
warped = imwarp(img, Rin, tform, 'OutputView', imref2d([6 7]), 'FillValues', fill);

save(fullfile(outDir, 'interp2d_probe.mat'), 'img', 'X', 'Y', 'out_linear', ...
    'fill', 'tform', 'warped', '-v7');
fprintf('wrote %s\n', fullfile(outDir, 'interp2d_probe.mat'));
end
