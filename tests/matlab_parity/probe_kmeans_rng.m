function probe_kmeans_rng()
% probe_kmeans_rng  Pin down the random-number consumption of the builtins
% MATLAB's kmeans relies on, so the Python port can replay the RNG stream.
%
%   * datasample(...,1,'Replace',false,'Weights',w) -> internal.stats.wswor
%     (compiled). Check: equals cumsum-inversion of ONE rand and consumes
%     exactly one number from the stream.
%   * randi(n) == floor(rand*n)+1 on the same stream.
%   * internal.stats.pdist2mex(...,'sqe') == explicit squared distances.
%   * internal.stats.groupcentroids == per-cluster mean.
%
% Run: matlab -batch "cd('<this dir>'); probe_kmeans_rng"
    N = 500;
    ok_idx = 0; ok_consume = 0; ok_randi = 0;
    for s = 1:N
        rng(s, 'twister'); w = rand(1, 7); w = w / sum(w); seq = rand(1, 3);
        rng(s, 'twister'); rand(1, 7);
        [~, i] = datasample((1:7)', 1, 'Replace', false, 'Weights', w);
        nxt = rand;
        edges = min([0 cumsum(w)], 1); edges(end) = 1;
        [~, ~, j] = histcounts(seq(1), edges);
        ok_idx = ok_idx + (i == j);
        ok_consume = ok_consume + (nxt == seq(2));
        rng(s, 'twister'); r = randi(1000); rng(s, 'twister'); r2 = floor(rand * 1000) + 1;
        ok_randi = ok_randi + (r == r2);
    end
    fprintf('wswor(k=1): idx == cumsum-inversion of one rand: %d/%d; consumed exactly one rand: %d/%d\n', ...
        ok_idx, N, ok_consume, N);
    fprintf('randi(n) == floor(rand*n)+1: %d/%d\n', ok_randi, N);

    rng(1, 'twister');
    X = rand(5, 100); C = rand(5, 3);
    D = internal.stats.pdist2mex(X, C, 'sqe', [], [], [], []);
    D2 = zeros(100, 3);
    for k = 1:3; D2(:, k) = sum((X - C(:, k)).^2, 1)'; end
    fprintf('pdist2mex sqe vs explicit: max abs diff %g\n', max(abs(D(:) - D2(:))));
    idx = randi(3, 100, 1);
    [Cg, ~] = internal.stats.groupcentroids(X, idx, 1:3, 'sqeuclidean', false);
    Cm = zeros(5, 3);
    for k = 1:3; Cm(:, k) = mean(X(:, idx == k), 2); end
    fprintf('groupcentroids vs mean: max abs diff %g\n', max(abs(Cg(:) - Cm(:))));
end
