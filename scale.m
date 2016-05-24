function [scaleddata,model] = scale( data )
sample_num=max(size(data));
model.max = max(data);
model.min = min(data);
maxmatrix= repmat(model.max,sample_num, 1);
minmatrix= repmat(model.min,sample_num, 1);
scaleddata=2*((data-minmatrix)./(maxmatrix-minmatrix))-1;
end

