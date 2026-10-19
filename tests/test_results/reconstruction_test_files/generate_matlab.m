% Run with CurveLab-2.1.2/fdct_wrapping_matlab first on the MATLAB path.
% Generates portable numerical references without microscopy image fixtures.
% Original transform routines: Laurent Demanet, 2004.
output_dir = fileparts(mfilename('fullpath'));
shapes = {[64 64], [96 128], [97 129], [128 96], [192 192], [65 80], [128 128], [128 128], [64 64], [64 64]};
coarse_angles = [16 16 16 16 16 16 8 32 16 16];
for case_id = 1:length(shapes)
    sz = shapes{case_id};
    input = reshape(mod((0:prod(sz)-1)*17,251)/7,sz);
    if case_id == 9, input = zeros(sz); end
    if case_id == 10, input = zeros(sz); input(19,37)=1; end
    scales = ceil(log2(min(sz))-3);
    angles = coarse_angles(case_id);
    C = fdct_wrapping(input,0,2,scales,angles);
    magnitudes = [];
    for j=1:length(C), for k=1:length(C{j}), magnitudes=[magnitudes;abs(C{j}{k}(:))];end;end
    magnitudes=sort(magnitudes,'descend');
    cutoff = magnitudes(round(0.2*length(magnitudes)));
    selected_scales = min(3,length(C)-1);
    Ct = C;
    for j=1:length(C),for k=1:length(C{j})
        Ct{j}{k}=C{j}{k}.*(abs(C{j}{k})>cutoff);
        if j<length(C)-selected_scales || j==length(C),Ct{j}{k}=zeros(size(C{j}{k}));end
    end;end
    reconstructed = real(ifdct_wrapping(Ct,0));
    roundtrip = ifdct_wrapping(C,0);
    matlab_version = version;
    forward_source = which('fdct_wrapping');
    inverse_source = which('ifdct_wrapping');
    save(fullfile(output_dir,sprintf('matlab_case_%02d.mat',case_id)), ...
        'selected_scales','input','C','Ct','cutoff','reconstructed','roundtrip','scales','angles', ...
        'matlab_version','forward_source','inverse_source','-v7');
end
