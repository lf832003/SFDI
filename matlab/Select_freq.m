% clear; 
clc;
load('/home/zs/SFDI_04172018/SFDI_training.mat'); % Change to the path to SFDI_training.mat
idx_permutation = [2, 3;
                   2, 4;
                   2, 5;
                   2, 6;
                   2, 7;
                   2, 8;
                   3, 4;
                   3, 5;
                   3, 6;
                   3, 7;
                   3, 8;
                   4, 5;
                   4, 6;
                   4, 7;
                   4, 8;
                   5, 6;
                   5, 7;
                   5, 8;
                   6, 7;
                   6, 8;
                   7, 8];
X_train = X_train';
y_train = y_train';
performance_mat = zeros(size(idx_permutation, 1), 1);
tr_cell = cell(size(idx_permutation, 1), 1);
for ii = 1:size(idx_permutation, 1)
    fprintf('Training frequencies pair %d.\n', ii);
    func_name = strcat('op_', num2str(ii));
    reflectance = X_train([1, idx_permutation(ii, :)], :);
    fitnet_script;
    performance_mat(ii) = performance;
    tr_cell{ii} = tr;
end
