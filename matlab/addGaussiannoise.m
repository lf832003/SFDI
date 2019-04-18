function noisyData = addGaussiannoise(reflectance)
sf = [0, 0.05, 0.1, 0.2, 0.4];
noiselevel = [0.0121, 0.0195, 0.0299, 0.0612, 0.1444];
sf_fit = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5];
p = polyfit(sf, noiselevel, 2);
noiselevel_fit = polyval(p, sf_fit);

noiseSigma = repmat(noiselevel_fit, size(reflectance, 1), 1).*reflectance;
noisyData = reflectance + noiseSigma.*randn(size(reflectance));