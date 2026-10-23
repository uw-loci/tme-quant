function call_original_bdc(pipeline, heDir, heFile, shgDir, ppm, outDir, goldenDir, curveAlignDir)
%CALL_ORIGINAL_BDC Registered H&E from the original CurveAlign function.
%
% Copies the case TIFFs into a temporary HE/SHG pair, calls BDcreation_reg
% (pipeline 'reg1') or BDcreation_reg2 ('reg2'), and writes
% outDir/registered_original.tif. The fixture tree is not modified.
% When goldenDir contains heFile, prints MAE against that committed golden.
%
% CurveAlign must be the directory argument or CURVEALIGN_MATLAB.
% BDcreation_reg.m does not seed kmeans; the caller says so. This function
% only records the file that function wrote.

    if strcmp(pipeline, 'reg1')
        requiredFile = 'BDcreation_reg.m';
    elseif strcmp(pipeline, 'reg2')
        requiredFile = 'BDcreation_reg2.m';
    else
        error('call_original_bdc:BadPipeline', ...
            'pipeline must be ''reg1'' or ''reg2'', got %s', pipeline);
    end

    curveAlignDir = resolve_curvealign_dir(curveAlignDir, requiredFile);
    addpath(curveAlignDir);

    heSrc = fullfile(heDir, heFile);
    shgSrc = fullfile(shgDir, heFile);
    if ~exist(heSrc, 'file')
        error('call_original_bdc:MissingInput', 'Missing HE %s', heSrc);
    end
    if ~exist(shgSrc, 'file')
        error('call_original_bdc:MissingInput', 'Missing SHG %s', shgSrc);
    end

    tmpRoot = tempname;
    heTmp = fullfile(tmpRoot, 'HE');
    shgTmp = fullfile(tmpRoot, 'SHG');
    [okHe, msgHe] = mkdir(heTmp);
    [okShg, msgShg] = mkdir(shgTmp);
    if ~okHe || ~okShg
        error('call_original_bdc:TempDir', ...
            'Could not create %s (%s) or %s (%s)', heTmp, msgHe, shgTmp, msgShg);
    end
    cleaner = onCleanup(@() remove_temp(tmpRoot)); %#ok<NASGU>

    copyfile(heSrc, fullfile(heTmp, heFile));
    copyfile(shgSrc, fullfile(shgTmp, heFile));

    params = struct( ...
        'HEfilepath', heTmp, ...
        'HEfilename', heFile, ...
        'pixelpermicron', ppm, ...
        'SHGfilepath', shgTmp);

    fprintf('  callOriginal: %s (%s) on temp copy %s\n', pipeline, requiredFile, tmpRoot);
    if strcmp(pipeline, 'reg1')
        BDcreation_reg(params);
    else
        BDcreation_reg2(params);
    end

    produced = fullfile(heTmp, 'HE_registered', heFile);
    if ~exist(produced, 'file')
        error('call_original_bdc:NoOutput', ...
            '%s did not write %s', requiredFile, produced);
    end
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    dest = fullfile(outDir, 'registered_original.tif');
    copyfile(produced, dest);
    fprintf('  saved %s\n', dest);

    if nargin >= 7 && ~isempty(goldenDir)
        goldenFile = fullfile(goldenDir, heFile);
        if ~exist(goldenFile, 'file')
            fprintf('  vs golden: missing %s\n', goldenFile);
        else
            report_registered_mae(dest, goldenFile);
        end
    end
end


function remove_temp(tmpRoot)
    if exist(tmpRoot, 'dir')
        rmdir(tmpRoot, 's');
    end
end


function report_registered_mae(registeredFile, goldenFile)
    registered = imread(registeredFile);
    golden = imread(goldenFile);
    if ~isequal(size(golden), size(registered))
        fprintf('  registered_original vs golden: SIZE MISMATCH %s vs %s\n', ...
            mat2str(size(golden)), mat2str(size(registered)));
        return;
    end
    delta = abs(double(golden) - double(registered));
    fprintf('  registered_original vs golden: MAE=%.4f exact=%.2f%% max=%d\n', ...
        mean(delta(:)), 100 * mean(delta(:) == 0), max(delta(:)));
end
