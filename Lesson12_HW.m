I1 = im2gray(imread('phily1.png'));
I2 = im2gray(imread('phily2.png'));
I3 = im2gray(imread('phily3.png'));
%I1 = (imread('phily1.png'));
%I2 = (imread('phily2.png'));
%I3 = (imread('phily3.png'));


images = cell(1,3);
images{1} = I1;
images{2} = I2;
images{3} = I3;

% Initialize all the transformations to the identity matrix. 
numImages = size(images,2);

% Initialize variable to hold image sizes.
imageSize = zeros(numImages,2); 

% Initialize variable to hold image transformations.
tforms(numImages) = projective2d(eye(3));


%%SURF Features
I_tmp = images{1};
% SIFT matching requires grayscale images.
if (size(I_tmp,3)>1)
    I_tmp = rgb2gray(I_tmp); % Convert to grayscale
end
imageSize(1,:) = size(I_tmp); % append image size.
% calculate features for first image
points1 = detectSURFFeatures(I_tmp);
[features, points] = extractFeatures(I_tmp,points1);

% Iterate over remaining image pairs
for n = 2:numImages
    fprintf('Processing image %d ... \n', n);
    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;
    I_tmp =  images{n}; % read nth image
    % SIFT matching requires grayscale images.
    if (size(I_tmp,3)>1)
        I_tmp = rgb2gray(I_tmp); % Convert to grayscale
    end
    imageSize(n,:) = size(I_tmp); % append image size.
    % Detect and extract SURF features for I(n).
    points = detectSURFFeatures(I_tmp);    
    [features, points] = extractFeatures(I_tmp, points);
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
    fprintf('- Number of matching features: %d\n', size(indexPairs,1));
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);        
    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estimateGeometricTransform2D(matchedPoints, matchedPointsPrev, 'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    % Compute T(1) * T(2) * ... * T(n-1) * T(n).
    tforms(n).T = tforms(n-1).T * tforms(n).T; 
end

for n = 1:numel(tforms)           
    [xlim(n,:), ylim(n,:)] = outputLimits(tforms(n), [1 imageSize(n,2)], [1 imageSize(n,1)]);    
end

avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);
fprintf('Center image is %d ... \n', centerImageIdx);

% apply the center image's inverse transformation to all the others.
Tinv = invert(tforms(centerImageIdx));
for n = 1:numel(tforms)    
    tforms(n).T = Tinv.T * tforms(n).T;
end

% generate panorama image
for n = 1:numel(tforms)           
    [xlim(n,:), ylim(n,:)] = outputLimits(tforms(n), [1 imageSize(n,2)], [1 imageSize(n,1)]);
end
 
maxImageSize = max(imageSize);
 
% Find the minimum and maximum output limits. 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);
 
yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);
width  = round(xMax - xMin);% Width of panorama.
height = round(yMax - yMin);% Height of panorama.
panorama = zeros([height width], 'like', I2); % Initialize "empty" panorama.

% create the panorama
blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');  
 
% Create a 2-D spatial reference object defining the size of the panorama.
panoramaView = imref2d([height width], [xMin xMax], [yMin yMax]);
 
% reconstruct image.
for n = 1:numImages
    I_tmp = images{n};   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I_tmp, tforms(n), 'OutputView', panoramaView);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I_tmp,1),size(I_tmp,2)), tforms(n), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
    imshow(panorama);
end
 
figure1 = figure;
% show result
subplot(2,2,1); imshow(images{1});title 'Philly1';
subplot(2,2,2); imshow(images{2});title 'Philly2';
subplot(2,2,3); imshow(images{3});title 'Philly3';
subplot(2,2,4); imshow(panorama);title 'Reconstructed';
saveas(figure1,'SURF Features');


%% Shift features 
I_tmp = images{1};
% SIFT matching requires grayscale images.
if (size(I_tmp,3)>1)
    I_tmp = rgb2gray(I_tmp); % Convert to grayscale
end
imageSize(1,:) = size(I_tmp); % append image size.
% calculate features for first image
points1 = detectSIFTFeatures(I_tmp);
[features, points] = extractFeatures(I_tmp,points1);

% Iterate over remaining image pairs
for n = 2:numImages
    fprintf('Processing image %d ... \n', n);
    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;
    I_tmp =  images{n}; % read nth image
    % SIFT matching requires grayscale images.
    if (size(I_tmp,3)>1)
        I_tmp = rgb2gray(I_tmp); % Convert to grayscale
    end
    imageSize(n,:) = size(I_tmp); % append image size.
    % Detect and extract SIFT features for I(n).
    points = detectSIFTFeatures(I_tmp);    
    [features, points] = extractFeatures(I_tmp, points);
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
    fprintf('- Number of matching features: %d\n', size(indexPairs,1));
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);        
    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estimateGeometricTransform2D(matchedPoints, matchedPointsPrev, 'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    % Compute T(1) * T(2) * ... * T(n-1) * T(n).
    tforms(n).T = tforms(n-1).T * tforms(n).T; 
end

for n = 1:numel(tforms)           
    [xlim(n,:), ylim(n,:)] = outputLimits(tforms(n), [1 imageSize(n,2)], [1 imageSize(n,1)]);    
end

avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);
fprintf('Center image is %d ... \n', centerImageIdx);

% apply the center image's inverse transformation to all the others.
Tinv = invert(tforms(centerImageIdx));
for n = 1:numel(tforms)    
    tforms(n).T = Tinv.T * tforms(n).T;
end

% generate panorama image
for n = 1:numel(tforms)           
    [xlim(n,:), ylim(n,:)] = outputLimits(tforms(n), [1 imageSize(n,2)], [1 imageSize(n,1)]);
end
 
maxImageSize = max(imageSize);
 
% Find the minimum and maximum output limits. 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);
 
yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);
width  = round(xMax - xMin);% Width of panorama.
height = round(yMax - yMin);% Height of panorama.
panorama2 = zeros([height width], 'like', I2); % Initialize "empty" panorama.

% create the panorama
blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');  
 
% Create a 2-D spatial reference object defining the size of the panorama.
panoramaView = imref2d([height width], [xMin xMax], [yMin yMax]);
 
% reconstruct image.
for n = 1:numImages
    I_tmp = images{n};   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I_tmp, tforms(n), 'OutputView', panoramaView);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I_tmp,1),size(I_tmp,2)), tforms(n), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama2 = step(blender, panorama2, warpedImage, mask);
    imshow(panorama2);
end
 
figure2 = figure;
% show result
subplot(2,2,1); imshow(images{1});title 'Philly1';
subplot(2,2,2); imshow(images{2});title 'Philly2';
subplot(2,2,3); imshow(images{3});title 'Philly3';
subplot(2,2,4); imshow(panorama2);title 'Reconstructed';
saveas(figure2,'SIFT Features');