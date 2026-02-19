function y = sampleVector(x, M)

    idx = randi(length(x), M, 1);  % random indices with replacement
    y = x(idx);                    % select elements from x
end

