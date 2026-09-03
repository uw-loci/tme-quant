function probe_es_steps(caseId)
% probe_es_steps  Reveal the (1+1)-ES first steps taken inside imregtform.
%
% With MaximumIterations = k the returned tform is the ES parent after k
% iterations. Compared with the identity start this exposes A * f_norm
% (radius scaling and the raw normal variates), which lets the Python port
% pin down the RNG seed and any per-pyramid-level radius refinement without
% access to the mex source. Results go to dumps/<caseId>/es_probe.txt.
%
% Run: matlab -batch "cd('<this dir>'); probe_es_steps('test1')"
    if nargin < 1; caseId = 'test1'; end
    thisDir = fileparts(mfilename('fullpath'));
    d = fullfile(thisDir, 'dumps', caseId);
    S = load(fullfile(d, 'images.mat'));
    HEmoving = S.HEmoving; fixedSHG = S.fixedSHG;

    outFile = fullfile(d, 'es_probe.txt');
    fid = fopen(outFile, 'w');
    cleanup = onCleanup(@() fclose(fid));

    [optimizer, metric] = imregconfig('multimodal');
    optimizer.InitialRadius = optimizer.InitialRadius / 3.5;

    % --- A: metric at (near-)identity: tiny radius, 1 iteration, 1 & 3 levels
    for levels = [1 3]
        o = optimizer; o.InitialRadius = 1e-12; o.MaximumIterations = 1;
        fprintf(fid, '### NEAR_IDENTITY levels=%d\n', levels);
        txt = evalc(['tf = imregtform(HEmoving, fixedSHG, ''similarity'', o, metric, ' ...
            '''PyramidLevels'', levels, ''DisplayOptimization'', true);']);
        fprintf(fid, '%s\n', txt);
        writeA(fid, 'tform_A', tf.A);
    end

    % --- B: first k ES iterations at a single pyramid level, several radii
    for tt = {'translation', 'similarity', 'affine'}
        for r = [optimizer.InitialRadius, 0.05]
            for k = [1 2 3 5]
                o = optimizer; o.InitialRadius = r; o.MaximumIterations = k;
                fprintf(fid, '### STEPS type=%s levels=1 radius=%.17g iters=%d\n', tt{1}, r, k);
                txt = evalc(['tf = imregtform(HEmoving, fixedSHG, tt{1}, o, metric, ' ...
                    '''PyramidLevels'', 1, ''DisplayOptimization'', true);']);
                fprintf(fid, '%s\n', txt);
                writeA(fid, 'tform_A', tf.A);
            end
        end
    end

    % --- C: three levels, 1 iteration each (per-level radius behaviour)
    for tt = {'translation', 'affine'}
        for k = [1 2]
            o = optimizer; o.InitialRadius = 0.05; o.MaximumIterations = k;
            fprintf(fid, '### STEPS type=%s levels=3 radius=%.17g iters=%d\n', tt{1}, o.InitialRadius, k);
            txt = evalc(['tf = imregtform(HEmoving, fixedSHG, tt{1}, o, metric, ' ...
                '''PyramidLevels'', 3, ''DisplayOptimization'', true);']);
            fprintf(fid, '%s\n', txt);
            writeA(fid, 'tform_A', tf.A);
        end
    end

    % --- D: same seed across calls? Two identical calls must give identical A.
    o = optimizer; o.InitialRadius = 0.05; o.MaximumIterations = 3;
    tf1 = imregtform(HEmoving, fixedSHG, 'affine', o, metric, 'PyramidLevels', 1);
    tf2 = imregtform(HEmoving, fixedSHG, 'affine', o, metric, 'PyramidLevels', 1);
    fprintf(fid, '### REPEAT_CALLS maxabsdiff=%.3g\n', max(abs(tf1.A(:) - tf2.A(:))));

    fprintf('wrote %s\n', outFile);
end

function writeA(fid, name, A)
    fprintf(fid, '%s =\n', name);
    for i = 1:size(A, 1)
        fprintf(fid, '  %.17g', A(i, :));
        fprintf(fid, '\n');
    end
end
