function d = arrdim(x)
%ARRDIM Human-style array dimensionality including cell contents
%   Scalars: 0
%   Vectors: 1
%   Matrices: 2
%   3D+: counts cell dimensions + content dimensions

    if iscell(x)
        % Check size of the cell array
        top_dims = find(size(x)>1, 1, 'last');
        if isempty(top_dims)
            top_dims = 0;   % 1x1 cell
        end
        
        % Check contents
        content_dims = zeros(size(x));
        for i = 1:numel(x)
            content_dims(i) = arrdim(x{i});
        end
        max_content_dim = max(content_dims(:));
        
        d = top_dims + max_content_dim;
        
    else
        % numeric, logical, char, etc.
        s = size(x);
        s = s(1:find(s>1,1,'last'));   % drop trailing singletons

        if isempty(s)
            d = 0;   % scalar
        elseif isvector(x)
            d = 1;
        else
            d = numel(s);
        end
    end
end
