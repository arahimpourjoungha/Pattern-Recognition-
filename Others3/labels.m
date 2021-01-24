function out = other_label(k,data)
out = [];
for i = 1:size(data,3)
    if  k~=i
        out = [out data(:,:,i)];
    end
end