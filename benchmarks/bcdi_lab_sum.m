function bcdi_lab_sum(alignedfile, supfile, threads)
% The summing half of the lab's support average, for bcdi_lab_matlab.py: the engine's own unring
% on every aligned pass-1 support, a sum with the first support counted twice (as the lab's driver
% does), cut at 20 % of the maximum. The engine folder, with its unring, is on the path.

maxNumCompThreads(threads);
a = load(alignedfile).aligned;
total = zeros(size(a, 1), size(a, 2), size(a, 3));
for k = 1:size(a, 4)
    u = unring(a(:, :, :, k));
    total = total + u;
    if k == 1, total = total + u; end
end
max_sum = max(total(:));
sup = uint8(total >= 0.2 * max_sum & total ~= 0);
voxels = nnz(sup);
save(supfile, 'sup', 'voxels', 'max_sum', '-v7');
fprintf('%s: %d voxels (max sum %.3f)\n', supfile, voxels, max_sum);
end
