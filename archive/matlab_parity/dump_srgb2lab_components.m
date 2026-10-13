function dump_srgb2lab_components()
% dump_srgb2lab_components  Export what makecform('srgb2lab') + applycform
% actually compute on uint8 input, so Python can evaluate it in closed form:
%   rgb/255 -> per-channel ICC TRC (spline-interpolated uint16 LUT)
%           -> [R;G;B colorant] matrix -> Bradford adaptation D50 -> ICC white
%           -> xyz2lab(wp = whitepoint('icc')) -> round([255 L/100, a+128, b+128])
% Since the input is uint8, the TRC is tabulated for the 256 possible values.
% Also writes a random uint8 RGB -> uint8 Lab reference set for testing, and
% checks that the reconstruction here reproduces applycform exactly.
%
% Run: matlab -batch "cd('<this dir>'); dump_srgb2lab_components"
    thisDir = fileparts(mfilename('fullpath'));
    outFile = fullfile(thisDir, 'dumps', 'srgb2lab_components.mat');

    c = makecform('srgb2lab');
    s2x = c.cdata.cforms{1};
    x2l = c.cdata.cforms{2};
    wp = x2l.cdata.whitepoint;
    if isfield(s2x.cdata, 'cforms')      % {mattrc, adapt}
        mattrc_c = s2x.cdata.cforms{1};
        adapt_c = s2x.cdata.cforms{2};
        adapter = adapt_c.cdata.adapter;
    else
        mattrc_c = s2x;
        adapter = eye(3);
    end
    MatTRC = mattrc_c.cdata.MatTRC;
    disp(MatTRC);
    colorants = [MatTRC.RedColorant; MatTRC.GreenColorant; MatTRC.BlueColorant];

    % applycurve (private): clamp, LUT/65535 on linspace(0,1,N), interp1 spline, clamp
    trcs = {MatTRC.RedTRC, MatTRC.GreenTRC, MatTRC.BlueTRC};
    x = (0:255)' / 255;
    trc256 = zeros(256, 3);
    for i = 1:3
        curve = trcs{i};
        if isstruct(curve)
            error('parametric TRC not handled by this dump');
        end
        if isa(curve, 'uint16'); scale = 65535; else; scale = 255; end
        if numel(curve) > 1
            lut1d = double(curve) / scale;
            samples = linspace(0, 1, numel(curve))';
            y = interp1(samples, lut1d, x, 'spline');
        else
            y = x .^ (double(curve) / 256);
        end
        trc256(:, i) = max(min(y, 1), 0);
    end

    % Reconstruct and verify against applycform on random uint8 triplets.
    rng(0, 'twister');
    n = 200000;
    rgb_u8 = uint8(floor(rand(n, 3) * 256));
    lab_ref = applycform(reshape(rgb_u8, n, 1, 3), c);
    lab_ref = reshape(lab_ref, n, 3);

    lin = zeros(n, 3);
    for i = 1:3
        lin(:, i) = trc256(double(rgb_u8(:, i)) + 1, i);
    end
    xyz = (lin * colorants) * adapter';
    xyzn = xyz ./ wp;
    f = xyzn .^ (1/3);
    L = xyzn <= 216 / 24389;
    f(L) = (841 / 108) * xyzn(L) + 16 / 116;
    lab = [116 * f(:, 2) - 16, 500 * (f(:, 1) - f(:, 2)), 200 * (f(:, 2) - f(:, 3))];
    lab_u8 = uint8(max(0, min(255, round([255 * lab(:, 1) / 100, lab(:, 2) + 128, lab(:, 3) + 128]))));
    nbad = nnz(any(lab_u8 ~= lab_ref, 2));
    fprintf('reconstruction vs applycform: %d / %d triplets differ\n', nbad, n);

    save(outFile, 'trc256', 'colorants', 'adapter', 'wp', 'rgb_u8', 'lab_ref', '-v7');
    fprintf('saved %s\n', outFile);
end
