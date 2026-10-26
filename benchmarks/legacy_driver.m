function legacy_driver(work_dir, n, seeds)
% Runs a lab MATLAB phasing engine on the files benchmarks/legacy.py writes, CPU only.
%
% The engine is not part of this repository; legacy.py puts its folders on the path. Every case
% file <name>.amp holds an n x n float64 measured amplitude, row-major. For each seed this writes
% <name>-<seed>.rec (real and imaginary parts, row-major, float64) and one line of
% <name>.times (seconds per run, after the first, which also loads the engine).
%
% The parameters are those of the lab's automatic BCDI phasing script, with the GPU off: 620
% iterations of ER and HIO (beta 0.9) switched at fixed iterations, a support cut to one quadrant
% at iteration 3, a half-size box start support, Gaussian shrinkwrap at threshold 0.15 in
% alternating windows with sigma going from 2 to 1, and a Gaussian detector mask widening from
% 0.7 to 1 of the array over the first half.

iterations = 620;
cases = dir(fullfile(work_dir, '*.amp'));
for c = 1:numel(cases)
    [~, name] = fileparts(cases(c).name);
    f = fopen(fullfile(work_dir, cases(c).name), 'r');
    data = reshape(fread(f, n * n, 'double'), n, n).';
    fclose(f);
    times = zeros(1, numel(seeds));
    for k = 1:numel(seeds)
        rng(seeds(k));
        params = struct();
        params.nn = [n, n, 1];
        params.sz = size(data);
        params.do_2D = 1;
        params.GPU = 0;
        params.GA = 0;
        params.pcdi = 0;
        params.silent = 1;
        params.iterations = iterations;
        params.start_guess = 'random-data';
        params.ALG1 = 'ERlr';
        params.ALG2 = 'HIOlr';
        params.trigger = [20, 200, 220, 400, 420, 600, 621];
        params.beta = 0.9;
        params.twin = 1;
        params.twin_it = 3;
        params.threshold = 0.15;
        params.sigma = 1;
        params.sw_type = 'gauss';
        params.shrink_trigger = round(make_vals(50, iterations, 1));
        params.phase_range = [-pi, pi];
        params.phase_trigger = [-1, -20];
        params.nonGA_lres_sig = make_vals(iterations / 2, 1, 2);
        params.nonGA_lres_det = make_vals(iterations / 2, 1, 0.7);
        params.mod_const_pow = 1;
        params.regularized_amp = 'none';
        params.no_zero = 0;
        params.iterate_avg_save = 0;
        params.save_every_iterate = 0;
        ss = round([0.5 * n, 0.5 * n, 1]);
        t = tic;
        [pn, support] = make_intial_support(data, ss, params);
        [pnm, support] = iterative_phasing_general(pn, data, support, params);
        times(k) = toc(t);
        out = pnm .* support;
        f = fopen(fullfile(work_dir, sprintf('%s-%d.rec', name, seeds(k))), 'w');
        fwrite(f, real(out).', 'double');
        fwrite(f, imag(out).', 'double');
        fclose(f);
    end
    f = fopen(fullfile(work_dir, [name, '.times']), 'w');
    fprintf(f, '%.4f\n', times);
    fclose(f);
end
end
