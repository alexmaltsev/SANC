function [centroiddata_restored, voldata_restored, voxellist_restored] = extract3DShape(centroiddata, voldata, voxellist, total_slices, noise_location)
    % Convert centroid data to a table and set variable names
    dataTable = array2table(centroiddata, 'VariableNames', {'X', 'Y', 'Z'});

    % Create the histogram and identify outlier slices
    [outlier_slice_indices, thresholded_slice_max] = identifyOutlierSlices(dataTable.Z, total_slices, noise_location);
    
    if (thresholded_slice_max > 0)

        % Filter data based on threshold
        [data_filtered, valid_indices] = filterData(dataTable, thresholded_slice_max);
    
        % Filter voldata and voxellist based on the same valid indices
        voldata_filtered = voldata(valid_indices);
        voxellist_filtered = voxellist(valid_indices);
    
        % Restore points based on spatial proximity
        [data_restored, restored_indices] = restorePoints(dataTable, data_filtered, outlier_slice_indices, thresholded_slice_max);
    
        % Convert table back to array for output and filter corresponding data
        centroiddata_restored = table2array(data_restored);
        voldata_restored = voldata(restored_indices);
        voxellist_restored = voxellist(restored_indices);
    else
        centroiddata_restored = centroiddata;
        voldata_restored = voldata;
        voxellist_restored = voxellist;
    end
end

function [outlier_slice_indices, thresholded_slice_max] = identifyOutlierSlices(zData, total_slices, noise_location)
    [counts, bin_edges] = histcounts(zData, total_slices);
    Q1 = prctile(counts, 25);
    Q3 = prctile(counts, 75);
    IQR = Q3 - Q1;
    threshold = Q3 + 3.0 * IQR;

    switch noise_location
        case 'start'
            outlier_slice_indices = find(counts > threshold & (1:numel(counts)) <= round(0.25 * numel(counts)));
        case 'end'
            outlier_slice_indices = find(counts > threshold & (1:numel(counts)) >= round(0.75 * numel(counts)));
    end
   
    if isempty(outlier_slice_indices)
        thresholded_slice_max = -1;
    else
        slice_shift = max(outlier_slice_indices) + 1;
        thresholded_slice_max = bin_edges(slice_shift + 1);
    end
end

function [data_filtered, valid_indices] = filterData(dataTable, thresholded_slice_max)
    if isnan(thresholded_slice_max)
        data_filtered = dataTable;
        valid_indices = (1:height(dataTable))';
    else
        valid_indices = find(dataTable.Z > thresholded_slice_max);
        data_filtered = dataTable(valid_indices, :);
    end
end

function [data_restored, restored_indices] = restorePoints(dataTable, data_filtered, outlier_slice_indices, thresholded_slice_max)
    if isempty(outlier_slice_indices)
        data_restored = data_filtered;
        restored_indices = (1:height(data_filtered))';
        return;
    end
    
    slice_shift = max(outlier_slice_indices) + 1;
    [counts_out, bin_edges_out] = histcounts(dataTable(dataTable.Z <= thresholded_slice_max, :).Z, slice_shift);

    data_restored = data_filtered;
    restored_indices = (1:height(data_filtered))';  % Initial indices from data_filtered

    for i = slice_shift:-1:min(outlier_slice_indices)
        bin_edge_out = bin_edges_out(i);
        next_bin_edge_out = bin_edges_out(i + 1);
        points_out = dataTable(dataTable.Z > bin_edge_out & dataTable.Z <= next_bin_edge_out, :);
        
        % Restore points based on proximity to previously restored data
        [data_restored, new_indices] = attemptRestore(points_out, data_restored, 1000);  % 1000 is the distance threshold
        restored_indices = [restored_indices; new_indices];
    end
end

function [data_restored, new_indices] = attemptRestore(points_out, data_restored, distance_threshold)
    % Compute pairwise distances and check proximity
    if isempty(points_out)
        new_indices = [];
        return;
    end
    
    distances = pdist2(table2array(points_out), table2array(data_restored));
    within_distance = any(distances < distance_threshold, 2);
    new_points = points_out(within_distance, :);
    new_indices = find(within_distance);
    
    data_restored = [data_restored; new_points];
end