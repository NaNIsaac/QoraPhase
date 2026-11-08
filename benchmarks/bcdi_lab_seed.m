function bcdi_lab_seed(casefile, outfile, stage, seed, supfile, threads, quick)
% Runs one seed of one pass of the lab's 3D BCDI engine (BCDI3d2) on CPU, for bcdi_lab_matlab.py.
%
% The engine is not part of this repository; bcdi_lab_matlab.py puts its folder on the path. The
% settings are those of the lab's two-pass driver. casefile holds raw (amplitude) and cut
% (amplitude with values <= 1 set to 0). stage 'pass1' starts from the engine's automatic support
% on cut, with shrinkwrap; 'pass2' keeps the support in supfile fixed, on raw. One seed per call
% gives the same result as the engine's own seed loop, which reseeds rng at every seed, and lets a
% long run stop between seeds. outfile gets the object, support, error per iteration, the seed's
% best-so-far error and the engine's wall time.

maxNumCompThreads(threads);
work = fileparts(outfile);
scratch = tempname(work);
mkdir(scratch);
d = load(casefile);
config = struct();
config.gen_num = 1;
config.methods = 'EHEHEHEHEHE';
config.run_num = [20 180 20 180 20 180 20 180 20 180 20];
config.lowres_f = [1 1 1 1 1 1 0 0 0 0 0];
config.shrink_para = [2, 0.1, 20];
config.lowres_para = [0.8, 2];
config.Ksplit = 1; config.rec_crit = 0.025; config.E_r = 1; config.G_M = 1;
config.min_itnum = 5;
config.bst = 0;
config.cmap_range = [0 0.5]; config.cmap = 'turbo';
config.mod = 'L2';
config.disp_flag = 'off'; config.GHIO_flag = 'off'; config.gpu_flag = 'off';
config.random_seeds = seed;
if strcmp(stage, 'pass1')
    config.dif_patt = d.cut;
    config.support = 'auto';
    config.shrink_f = [1 1.1 1.2 1.3 1.4 1.5 0 0 0 0 0];
else
    s = load(supfile);
    config.dif_patt = d.raw;
    config.support = double(s.sup);
    config.shrink_f = zeros(1, 11);
end
if quick
    config.run_num = 4 * ones(1, 11);
    config.shrink_para(3) = 4;
end
config.sv_size = size(config.dif_patt);
config.save_name = fullfile(scratch, 'bed.mat');
here = pwd;
cd(scratch);
save('config.mat', 'config', '-v7.3');
t = tic;
BCDI3d2
wall = toc(t);
r = load(config.save_name).sv_bed(1).results(1);
cd(here);
rmdir(scratch, 's');
obj = single(r.r_space);
sup = r.support > 0;
err = r.err_array(:).';
best_e = r.best_e(1);
save(outfile, 'obj', 'sup', 'err', 'best_e', 'seed', 'wall', '-v7');
fprintf('%s seed %d: engine %.1f s, threads %d, MATLAB %s\n', stage, seed, wall, maxNumCompThreads, version);
end
