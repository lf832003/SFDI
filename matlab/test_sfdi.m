function [mse, abs_error, rel_error] = test_sfdi(X_test, y_test)
X_test = X_test';
y_test = y_test';

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
mse = zeros(size(idx_permutation, 1), 2);
abs_error = zeros(size(idx_permutation, 1), 2);
rel_error = zeros(size(idx_permutation, 1), 2);
for ii = 1:size(idx_permutation, 1)
    y_pred = sfdi_wrapper(strcat('op_', num2str(ii)), ...
        X_test([1, idx_permutation(ii, :)], :));
    mse(ii, 1) = immse(y_test(1, :), y_pred(1, :));
    mse(ii, 2) = immse(y_test(2, :), y_pred(2, :));
    abs_error(ii, 1) = mean(abs(y_test(1, :) - y_pred(1, :)));
    abs_error(ii, 2) = mean(abs(y_test(2, :) - y_pred(2, :)));
    rel_error(ii, 1) = mean(abs(y_test(1, :) - y_pred(1, :))./y_test(1, :));
    rel_error(ii, 2) = mean(abs(y_test(2, :) - y_pred(2, :))./y_test(2, :));
end
    
