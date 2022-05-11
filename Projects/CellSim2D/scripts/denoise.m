for i = 1 : 50
    filename = strcat("/home/yueli/Documents/ETH/WuKong/output/FilteredData/", int2str(i-1), ".txt");
    data = load(filename);
    data_point_cloud = pointCloud(data);
    [ptCloudOut,inlierIndices,outlierIndices] = pcdenoise(data_point_cloud);
    denoised_data = ptCloudOut.Location;
    [row col] = size(data);
    filename_denoised = strcat("/home/yueli/Documents/ETH/WuKong/output/FilteredData/", int2str(i-1), "_denoised.txt");
    result = zeros(row,col)-1;
    for j = 1:length(inlierIndices)
        result(inlierIndices(j), :) = denoised_data(j, :);
    end
    writematrix(result, filename_denoised,'Delimiter','space');
end