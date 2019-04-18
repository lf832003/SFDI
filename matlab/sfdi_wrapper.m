function y = sfdi_wrapper(func_name, x)
% x num_of_reflectance times num_of_samples
y = feval(func_name, x);
y(2, :) = y(2, :) * 10.0;