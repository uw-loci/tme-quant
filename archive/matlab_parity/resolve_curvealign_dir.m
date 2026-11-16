function curveAlignDir = resolve_curvealign_dir(curveAlignDir, requiredFile)
%RESOLVE_CURVEALIGN_DIR Folder that contains a CurveAlign BDcreation script.
%
% curveAlignDir is the argument passed to dump_bdc_reg1 / dump_bdc_reg2.
% When that argument is empty, CURVEALIGN_MATLAB is used. requiredFile is
% 'BDcreation_reg.m' or 'BDcreation_reg2.m'. The error names the path that
% was checked. This repo does not vendor those scripts.

    if nargin < 1 || isempty(curveAlignDir)
        curveAlignDir = strtrim(getenv('CURVEALIGN_MATLAB'));
    end
    if isempty(curveAlignDir)
        error('resolve_curvealign_dir:NotSet', ...
            ['Pass the CurveAlign directory or set CURVEALIGN_MATLAB. ' ...
             'That folder must contain %s.'], requiredFile);
    end
    if ~exist(curveAlignDir, 'dir')
        error('resolve_curvealign_dir:MissingDir', ...
            'CurveAlign directory not found: %s', curveAlignDir);
    end
    scriptPath = fullfile(curveAlignDir, requiredFile);
    if ~exist(scriptPath, 'file')
        error('resolve_curvealign_dir:MissingScript', ...
            'Expected %s', scriptPath);
    end
end
