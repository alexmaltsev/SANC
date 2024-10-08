function rc = processRyRClusterTIFFs(inputFolder, outputFolder, species, export_tif)
    % Check if the output folder exists, create if not
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    % List all TIF files in the input directory
    tiffFiles = dir(fullfile(inputFolder, '*.tif'));
    
    for i = 1:length(tiffFiles)
        filePath = fullfile(inputFolder, tiffFiles(i).name);
        fprintf('Processing %s...\n', filePath);

        % Call processSingleVolume for each TIF file, passing the output folder
        rc = processSingleVolume(tiffFiles(i).name(1:end-4), filePath, species, outputFolder, export_tif);
        
        if rc ~= 0
            fprintf('Error processing %s\n', filePath);
        end
    end
    
    fprintf('All files processed.\n');
    rc = 0; % Return zero if the function executed successfully
end

function rc = processSingleVolume(cell_name, f, species, outputFolder, export_tif)
    % Define voxel dimensions based on the species
    [vox_x, vox_y, vox_z] = getVoxelDimensions(species);
    
    % Read the images into a 3D volume
    myVol = readAndConstructVolume(f);

    % Process the 3D volume to get centroids and other metrics
    [centroiddata, voldata, voxellist] = processVolume(myVol, vox_x, vox_y, vox_z);
    [centroiddata, voldata, voxellist] = extract3DShape(centroiddata, voldata, voxellist, size(myVol, 3), 'start');
   
    % Compute clusters and alpha shape, filter out centroids not part of the largest cluster
    [B, A, shp, cluster_labels, centroiddata, voldata, voxellist] = findAlphaShape(centroiddata, voldata, voxellist);

    % Define the folder to save results, using the outputFolder parameter
    folder_name = fullfile(outputFolder, cell_name);
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end
    
    % Rebuild the filtered volume using the centroids that are part of the alpha shape
    if strcmp(export_tif, 'true')
        filteredVol = rebuildVolume(myVol, voxellist, centroiddata);
        exportFilteredVolume(filteredVol, folder_name);
    end

    saveFiguresAndData(B, A, folder_name, shp, centroiddata, voldata, voxellist, cluster_labels);
    
    % Return zero if the function executed successfully
    rc = 0;
end


function [B, A, shp, cluster_labels, centroiddata, voldata, voxellist] = findAlphaShape(centroiddata, voldata, voxellist)
    % Determine epsilon using median nearest neighbor distances
    scalar = 3;
    c = -1/(sqrt(2)*erfcinv(3/2));
    B0 = calculateNearestNeighbors(centroiddata);
    mad = median(abs(B0-median(B0)));
    eps = median(B0) + scalar*c*mad;  
    min_samples = 3;

    % Perform DBSCAN clustering
    cluster_labels = dbscan(centroiddata, eps, min_samples);

    % Identify the largest cluster
    [unique_clusters, cluster_sizes] = groupClusters(cluster_labels);
    if isempty(unique_clusters)
        largest_cluster_label = NaN;  
        boundary_indices = [];
    else
        [~, idx] = max(cluster_sizes);
        largest_cluster_label = unique_clusters(idx);
        largest_cluster_indices = find(cluster_labels == largest_cluster_label);
        largest_cluster_points = centroiddata(largest_cluster_indices, :);
                
        % Compute alpha shape for the largest cluster
        shp = alphaShape(largest_cluster_points(:,1), largest_cluster_points(:,2), largest_cluster_points(:,3));
        initialAlpha = 10000; %make higher and lower
        
        % Define the range of alpha multipliers
        alphaMultipliers = linspace(0.1, 1, 250); % 100 steps from 0.2 to 1 times the initialAlpha
        volumes = zeros(length(alphaMultipliers), 1);
        areas = zeros(length(alphaMultipliers), 1);
        densities = zeros(length(alphaMultipliers), 1);
        
        % Number of points in the cluster
        numPoints = size(largest_cluster_points, 1);
        
        % Iterate through each alpha multiplier
        for i = 1:length(alphaMultipliers)
            shp.Alpha = initialAlpha * alphaMultipliers(i);
            volumes(i) = volume(shp); % Compute and store the volume
            areas(i) = surfaceArea(shp); % Compute and store the surface area
            densities(i) = numPoints / areas(i); % Compute and store the density
        end
        
        % Find the index of the maximum density
        [~, maxDensityIdx] = max(densities);
        
        % Determine the optimal alpha value corresponding to the peak in density
        optimalAlpha = initialAlpha * alphaMultipliers(maxDensityIdx);
                
        % Set the alpha of the alpha shape to the optimal alpha value
        shp.Alpha = optimalAlpha;
        
        % Display the optimal alpha value
        disp(['Optimal Alpha Value: ', num2str(optimalAlpha)]);
        
        % Construct adjacency matrix from alpha shape
        [bf, P] = boundaryFacets(shp);
        
        % Filter surface objects
        idx = knnsearch(centroiddata, P);
        centroiddata = centroiddata(idx, :);
        voxellist = voxellist(idx);
        voldata = voldata(idx);
        
        % Align unique points with centroiddata
        [uniquePoints, ~, ic] = unique(P, 'rows', 'stable');
        N = size(uniquePoints, 1);
        A = zeros(N, N);
        
        % Create mapping from uniquePoints to centroiddata
        [~, idxMap] = ismember(uniquePoints, centroiddata, 'rows');
        
        % Calculate distances between unique points
        for x = 1:size(bf, 1)
            tri = bf(x, :);
            for pair = nchoosek(tri, 2)'  % Generate combinations of 2 points
                i = idxMap(ic(pair(1)));
                j = idxMap(ic(pair(2)));
                dist = sqrt(sum((centroiddata(i, :) - centroiddata(j, :)).^2));
                A(i, j) = dist;
                A(j, i) = dist;
            end
        end

        % Calculate nearest neighbor distances
        B = zeros(N,1);
        for i = 1:N
            distances = A(i, :);
            distances(distances == 0) = inf;  % Ignore self-distances
            B(i) = min(distances);
        end
    end
end


function [vox_x, vox_y, vox_z] = getVoxelDimensions(species)
    switch species
        case 'rabbit'
            vox_x = 55.5; vox_y = 55.5; vox_z = 150;
        case 'mouse'
            vox_x = 100; vox_y = 100; vox_z = 100;
    end
end

function myVol = readAndConstructVolume(f)
    info = imfinfo(f);
    numImages = numel(info);
    X = cell(1, numImages);
    for k = 1:numImages
        X{k} = imread(f, k, 'Info', info);
    end
    myVol = cat(3, X{:});
end

function centroiddata_scaled = scaleCentroids(centroiddata_unscaled, vox_x, vox_y, vox_z)
    centroiddata_scaled = centroiddata_unscaled .* [vox_x, vox_y, vox_z];
end

function B = calculateNearestNeighbors(centroiddata)
    % Get the number of centroids
    len = size(centroiddata,1);
    B = zeros(len,1);
    
    for i = 1:len
        current_point = centroiddata(i,:);
        distance_search = centroiddata-repmat(current_point,[len 1]);
        dist_from_current_point = sqrt((distance_search(:,1)).^2+(distance_search(:,2).*(1)).^2+(distance_search(:,3).*(1)).^2);
        dist_from_current_point(dist_from_current_point <= 0)= [];
        nearest_dist = min(dist_from_current_point);
        B(i) = nearest_dist;
    end
    
end

function [unique_clusters, cluster_sizes] = groupClusters(cluster_labels)
    unique_clusters = unique(cluster_labels);
    cluster_sizes = histc(cluster_labels, unique_clusters);
end

function [centroiddata, voldata, voxellist] = processVolume(myVol, vox_x, vox_y, vox_z)
    stats = regionprops3(myVol, 'Volume', 'Centroid', 'VoxelList');
    stats(any(ismissing(stats), 2), :) = [];
    stats(1, :) = [];
    centroiddata = scaleCentroids(table2array(stats(:, 'Centroid')), vox_x, vox_y, vox_z);
    voldata = table2array(stats(:, 'Volume'));
    voxellist = table2array(stats(:, 'VoxelList'));
end

function filteredVol = rebuildVolume(originalVol, voxellist, centroiddata)
    filteredVol = zeros(size(originalVol), 'uint16');
    label = 0;
    for i = 1:length(centroiddata)
        label = label + 1;  % Increment label for each object
        for j = 1:size(voxellist{i}, 1)
            x = voxellist{i}(j, 2);
            y = voxellist{i}(j, 1);
            z = voxellist{i}(j, 3);
            filteredVol(x, y, z) = label;  % Assign a unique label to each voxel of the object
        end
    end
end

function exportFilteredVolume(filteredVol, folder_name)
    % Extract the last folder name to use in filenames
    [~, last_folder_name] = fileparts(folder_name);

    % Define file path for the export using the last folder name
    outputFilePath = fullfile(folder_name, sprintf('%s_filtered_volume.tif', last_folder_name));

    % Export each slice of the volume
    for k = 1:size(filteredVol, 3)
        if k == 1
            imwrite(uint16(filteredVol(:, :, k)), outputFilePath, 'WriteMode', 'overwrite', 'Compression', 'none');
        else
            imwrite(uint16(filteredVol(:, :, k)), outputFilePath, 'WriteMode', 'append', 'Compression', 'none');
        end
    end
end

function saveFiguresAndData(B, A, folder_name, shp, centroiddata, voldata, voxellist, cluster_labels)
    % Extract the last folder name to use in filenames
    [~, last_folder_name] = fileparts(folder_name);

    % Save figures and data as before
    fig_name = fullfile(folder_name, sprintf('%s_alphashape.fig', last_folder_name));
    h = figure('visible', 'off');
    plot(shp, 'FaceColor', 'green', 'FaceAlpha', 0.10, 'EdgeColor', 'black');
    axis equal;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Alpha Shape with Transparent Faces');
    savefig(h, fig_name);
    jpg_name = fullfile(folder_name, sprintf('%s_alphashape.jpg', last_folder_name));
    saveas(h, jpg_name);
    close(h);

    % Save surface area and volume
    sa_filename = fullfile(folder_name, sprintf('%s_surfacearea.csv', last_folder_name));
    volume_filename = fullfile(folder_name, sprintf('%s_volume.csv', last_folder_name));
    sa = surfaceArea(shp);
    vol = volume(shp);
    dlmwrite(sa_filename, sa, 'delimiter', ',');
    dlmwrite(volume_filename, vol, 'delimiter', ',');

    % Save centroid data, volume data, and other arrays
    centroids_filename = fullfile(folder_name, sprintf('%s_centroids.csv', last_folder_name));
    volumes_filename = fullfile(folder_name, sprintf('%s_volumes.csv', last_folder_name));
    voxel_list_filename = fullfile(folder_name, sprintf('%s_voxellist.csv', last_folder_name));
    clusters_filename = fullfile(folder_name, sprintf('%s_clusterlabels.csv', last_folder_name));
    adj_matrix_filename = fullfile(folder_name, sprintf('%s_adjmatrix.csv', last_folder_name));
    nearest_neighbors_filename = fullfile(folder_name, sprintf('%s_neighboralphacentroid.csv', last_folder_name));

    mesh_filename = fullfile(folder_name, sprintf('%s_alphashape.obj', last_folder_name));
    exportAlphaShapeMesh(shp, centroiddata, voldata, mesh_filename);

    dlmwrite(centroids_filename, centroiddata, 'delimiter', ',');
    ma = [centroiddata,voldata];
    dlmwrite(volumes_filename, ma, 'delimiter', ',');
    writetable(cell2table(voxellist), voxel_list_filename);
    dlmwrite(clusters_filename, cluster_labels, 'delimiter', ',');
    dlmwrite(adj_matrix_filename, A, 'delimiter', ',');
    dlmwrite(nearest_neighbors_filename, B, 'delimiter', ',');

    % Optionally, print a confirmation that files have been saved
    disp(['Files have been saved in ', folder_name]);
end

function [B, A, P] = calculateNearestNeighborsAndAdjacencyMatrix(bf, P)
    N = length(P);
    A = zeros(N, N);
    B = zeros(N, 1);

    % Calculate adjacency matrix and nearest neighbor distances
    for x = 1:length(bf)
        tri = bf(x,:);
        for j = 1:3
            P1 = P(tri(j), :);
            P2 = P(tri(mod(j, 3) + 1), :);
            dist = sqrt(sum((P1 - P2).^2));
            A(tri(j), tri(mod(j, 3) + 1)) = dist;
            A(tri(mod(j, 3) + 1), tri(j)) = dist;  % Ensure the matrix is symmetric
        end
    end

    % Calculate nearest neighbor distances
    for i = 1:N
        distances = A(i, :);
        distances(distances == 0) = inf;  % Ignore self-distances
        B(i) = min(distances);
    end
end

function exportAlphaShapeMesh2(shp, centroiddata, outputFilePath)
    % Construct adjacency matrix from alpha shape
    [bf, P] = boundaryFacets(shp);

    % Filter surface objects
    idx = knnsearch(centroiddata, P);
    centroiddata = centroiddata(idx, :);

    % Align unique points with centroiddata
    [uniquePoints, ~, ic] = unique(P, 'rows', 'stable');

    % Create mapping from uniquePoints to centroiddata
    [~, idxMap] = ismember(uniquePoints, centroiddata, 'rows');

    % Map the faces to the new vertex indices
    mappedFaces = reshape(idxMap(ic(bf)), size(bf));

    % Write the mesh to an OBJ file
    objFile = fopen(outputFilePath, 'w');

    if objFile == -1
        error('Failed to open the output file for writing.');
    end

    % Write vertices (filtered centroiddata)
    fprintf(objFile, 'o AlphaShape\n');
    for i = 1:size(centroiddata, 1)
        fprintf(objFile, 'v %f %f %f\n', centroiddata(i, 1), centroiddata(i, 2), centroiddata(i, 3));
    end

    % Write faces (mapped to filtered centroid indices)
    for i = 1:size(mappedFaces, 1)
        fprintf(objFile, 'f %d %d %d\n', mappedFaces(i, 1), mappedFaces(i, 2), mappedFaces(i, 3));
    end

    fclose(objFile);

    % Validate the exported OBJ file
    if exist(outputFilePath, 'file') == 2
        disp(['OBJ file exported successfully: ', outputFilePath]);
    else
        error('Failed to export the OBJ file.');
    end
end

function exportAlphaShapeMesh(shp, centroiddata, voldata, outputFilePath)
    % Construct adjacency matrix from alpha shape
    [bf, P] = boundaryFacets(shp);

    % Filter surface objects
    idx = knnsearch(centroiddata, P);
    centroiddata = centroiddata(idx, :);
    voldata = voldata(idx);

    % Align unique points with centroiddata
    [uniquePoints, ~, ic] = unique(P, 'rows', 'stable');

    % Create mapping from uniquePoints to centroiddata
    [~, idxMap] = ismember(uniquePoints, centroiddata, 'rows');

    % Map the faces to the new vertex indices
    mappedFaces = reshape(idxMap(ic(bf)), size(bf));

    % Write the mesh to an OBJ file
    objFile = fopen(outputFilePath, 'w');

    if objFile == -1
        error('Failed to open the output file for writing.');
    end

    % Write vertices (filtered centroiddata)
    fprintf(objFile, 'o AlphaShape\n');
    for i = 1:size(centroiddata, 1)
        fprintf(objFile, 'v %f %f %f\n', centroiddata(i, 1), centroiddata(i, 2), centroiddata(i, 3));
    end

    % Write vertex radii as custom vertex attributes
    fprintf(objFile, '\n');
    for i = 1:size(voldata, 1)
        radius = nthroot((3/4) * voldata(i) / pi, 3);
        fprintf(objFile, 'vt %f\n', radius);
    end

    % Write faces (mapped to filtered centroid indices)
    fprintf(objFile, '\n');
    for i = 1:size(mappedFaces, 1)
        fprintf(objFile, 'f %d/%d %d/%d %d/%d\n', mappedFaces(i, 1), mappedFaces(i, 1), mappedFaces(i, 2), mappedFaces(i, 2), mappedFaces(i, 3), mappedFaces(i, 3));
    end

    fclose(objFile);

    % Validate the exported OBJ file
    if exist(outputFilePath, 'file') == 2
        disp(['OBJ file exported successfully: ', outputFilePath]);
    else
        error('Failed to export the OBJ file.');
    end
end