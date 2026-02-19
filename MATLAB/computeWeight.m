function w1 = computeWeight(mu1,mu2,k, sigma1, sigma2, sigma12)
    EX2 = sigma1+sigma2-2*sigma12+mu1^2+mu2^2-2*mu1*mu2;
    w1 = ((mu1-mu2)/2+k*(sigma2+mu2^2-sigma12-mu1*mu2))/(k*EX2)
end
