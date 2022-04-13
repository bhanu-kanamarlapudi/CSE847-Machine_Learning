data = load('USPS.mat');
data = data.A;
p = [10, 50, 100, 200];% number of principal components
num_p = size(p,2);
index = randsample(size(data,1),1);% randomly selecting image
fprintf("Random selected Index - %d \n",index);
img_orig = reshape(data(index,:), 16, 16);
subplot(1,num_p+1,1); 
imshow(img_orig');% displaying randomly selected image
title('Original Image');
A_copy = data;
for i=1:num_p
    fprintf('Number of principal components - %d\n',p(i));
    data= A_copy - mean(A_copy); % centering data
    [U, S, V] = svd(data); % computing svd
    data = data * V(:,1:p(i));
    recon_data = data*V(:, 1:p(i))'; % reconstructing image
    recon_error = norm(A_copy - recon_data);
    fprintf("The Reconstruction Error is %f \n",recon_error);
    img_recon = reshape(recon_data(index,:), 16, 16);
    subplot(1,num_p+1,i+1); 
    imshow(img_recon'); % displaying reconstructed image
    title(strcat('p=', int2str(p(i)))); 
end