function probe_reg1_kmeans(caseId, nSeeds)
% probe_reg1_kmeans  Enumerate the distinct k-means optima BDcreation_reg.m can
% land in (it never seeds the RNG) and run the full reg1 pipeline once per
% optimum, reporting each result against the committed golden.
%
% Run: matlab -batch "cd('<this dir>'); probe_reg1_kmeans('test8', 40)"
% Needs dumps/<caseId>/intermediates.mat from dump_bdc_reg1 first (for `ab`).
    if nargin < 1; caseId = 'test8'; end
    if nargin < 2; nSeeds = 40; end
    thisDir = fileparts(mfilename('fullpath'));
    S = load(fullfile(thisDir, 'dumps', caseId, 'intermediates.mat'), 'lab_HEdata');
    ab = double(S.lab_HEdata(:,:,2:3));
    [nrows, ncols, ~] = size(ab);
    ab = reshape(ab, nrows * ncols, 2);

    masks = {};
    reps = [];
    counts = [];
    for s = 0:nSeeds-1
        rng(s, 'twister');
        [ci, cc] = kmeans(ab, 3, 'distance', 'sqEuclidean', 'Replicates', 3);
        [~, ix] = sort(mean(cc, 2));
        mask = reshape(ci, nrows, ncols) == ix(3);
        found = 0;
        for k = 1:numel(masks)
            if isequal(masks{k}, mask); found = k; break; end
        end
        if found
            counts(found) = counts(found) + 1;
        else
            masks{end+1} = mask; %#ok<AGROW>
            reps(end+1) = s;     %#ok<AGROW>
            counts(end+1) = 1;   %#ok<AGROW>
        end
    end
    fprintf('\n%d distinct collagen masks over %d seeds:\n', numel(masks), nSeeds);
    for k = 1:numel(masks)
        fprintf('  optimum %d: representative seed %d, hit %d/%d times, %d px\n', ...
            k, reps(k), counts(k), nSeeds, nnz(masks{k}));
    end
    fprintf('\nRunning the full reg1 pipeline once per optimum:\n');
    for k = 1:numel(masks)
        dump_bdc_reg1(caseId, false, reps(k), sprintf('_km%d', k));
    end
end
