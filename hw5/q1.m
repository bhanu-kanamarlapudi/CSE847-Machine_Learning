data = importdata('housing.csv');
data = data.data;
data = data(:,[1 7 8 9]);
num_clusters = [4 5 6 7 8];
sse_kmeans = zeros(1, size(num_clusters,2));
sse_kmeans_spectral = zeros(1, size(num_clusters,2));
num_samples = size(data,1);
% init_cluster_assignments = zeros(num_samples, 1);
for i=1:size(num_clusters,2)
    [cluster_assignments] = kmeans_(data,init_cluster_assignments, num_clusters(i), num_samples);
    sse_kmeans(i) = calc_sse(data, cluster_assignments, num_clusters(i));
    kmeansplot(data, cluster_assignments, num_clusters(i));
    [cluster_assignments] = kmeans_spectral(data, init_cluster_assignments, num_clusters(i), num_samples);
    sse_kmeans_spectral(i) = calc_sse(data, cluster_assignments, num_clusters(i));
    kmeansplot(data, cluster_assignments, num_clusters(i));
end
figplot(num_clusters, sse_kmeans)
figplot(num_clusters, sse_kmeans_spectral)

function [] = figplot(num_clusters, sse)
figure; 
plot(num_clusters, sse); 
xlabel('Num Clusters'); 
ylabel('SSE'); 
title('Variation of SSE');
end

function [cluster_assignments] = kmeans_spectral(data, cluster_assignments, num_cluster,num_samples)
   [U, S, V] = svd(data);
   projection = U(:, 1:num_cluster);
   rand_mat = rand(num_cluster,num_cluster);
   orth_mat = orth(rand_mat);
   data = projection * orth_mat;
   cluster_assignments= kmeans_(data, cluster_assignments, num_cluster,num_samples);
end

function [cluster_assignments] = kmeans_(data, cluster_assignments, num_cluster, num_samples) 
    temp = randperm(num_samples);
    cluster_center_idx = temp(1:num_cluster);
    cluster_centers = data(cluster_center_idx, :);
    fprintf('Initialized Cluster centers - %d\n', cluster_centers);
    diff = inf;
    count_iter = 0;
    while(diff ~= 0)
        count_iter = count_iter+1;
        prev_cluster_assignments = cluster_assignments;
        % for each sample find the cluster which is at minimum distance
        for sample=1:num_samples
            min_distance = inf;
            min_index = -1;
            for cluster_index = 1:num_cluster
                cur_distance = norm(data(sample,:) - cluster_centers(cluster_index,:));
                if(cur_distance < min_distance)
                    min_distance = cur_distance;
                    min_index = cluster_index;
                end
            end
            cluster_assignments(sample,1) = min_index;
        end
        
        for cluster_index = 1:num_cluster
            cluster_centers(cluster_index,:) = mean(data(cluster_assignments == cluster_index,:));
        end
        
        diff = sum(prev_cluster_assignments ~= cluster_assignments);
    end
    SSE = calc_sse(data, cluster_assignments, num_cluster);
    fprintf('SSE when number of clusters=%d is %f\n', num_cluster, SSE);
    
end

function [sse] = calc_sse(data, cluster_assignments, num_clusters)
    sse = 0;
    for cluster = 1:num_clusters
        cluster_data = data(cluster_assignments == cluster,:);
        cluster_center = mean(cluster_data);
        sse = sse + norm(cluster_data-cluster_center);
    end
end

function [] = kmeansplot(data, cluster_assignments, num_clusters)
    num_features = size(data,2);
    if num_features>2
        pc = pca(data);
        data = data * pc(:, 1:2);
    end
    figure;
    hold on;
    gscatter(data(:,1),data(:,2),cluster_assignments);
    title(strcat('Num clusters=', int2str(num_clusters)));
    hold off;
    pause(3);
end
