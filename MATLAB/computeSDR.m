function sdr = computeSDR(A,B,R)

    DA=R-A;
    DB=R^2-B; 
    denom = B - A^2;
    if denom<1e-8
        fprintf('True')
    end
    denom = max(denom, 1e-8);
    sdr=(B*DA-0.5*A*DB)/(denom)^(3/2);
end