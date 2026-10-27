function measured_driver(param_file, data_file, out_dir, seeds)
% Reruns a lab MATLAB phasing engine on a measured dataset with the parameters of one of its saved
% reconstructions, once per seed, for benchmarks/measured.py --lab.
%
% The engine is not part of this repository; the caller puts its folders on the path. param_file
% is the engine's saved -PARAMS.mat, data_file its prepared intensity (array). Each seed writes
% lab-<seed>.mat: the last iterate (after the Fourier step, before the support step), the final
% support, the engine's chi and the seconds taken. Seeds whose file exists are skipped, so an
% interrupted run resumes.
saved = load(param_file).params;
% Fields the engine writes into the struct it returns; the rest are its inputs.
outputs = {'itno', 'pn', 'temp_error', 'prev_coh', 'final_chi', 'elapsed_time'};
saved = rmfield(saved, intersect(outputs, fieldnames(saved)));
saved.update = 1;
saved.silent = 1;
saved.coh = gauss_3D(11, 11, 11, saved.kernalxy(1), saved.kernalxy(2), saved.kernalxy(3));
data = sqrt(abs(double(load(data_file).array)));
ss = round(saved.size_init([2, 1, 3]));
for seed = seeds
    out = fullfile(out_dir, sprintf('lab-%d.mat', seed));
    if isfile(out)
        fprintf('seed %d: kept\n', seed);
        continue
    end
    rng(seed);
    t = tic;
    [pn, support] = make_intial_support(data, ss, saved);
    [pnm, support, params_out] = iterative_phasing_general(pn, data, support, saved);
    seconds = toc(t);
    iterate = single(gather(pnm));
    support = logical(gather(support));
    chi = double(gather(params_out.final_chi));
    save([out, '.part'], 'iterate', 'support', 'chi', 'seconds', '-v7');
    movefile([out, '.part'], out);
    fprintf('seed %d: chi %.4f, %.1f s\n', seed, chi, seconds);
end
end
