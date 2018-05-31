function logP = log_partition(rbm_w)
    [hid_units, vis_units] = size(rbm_w);
    PS = dec2bin(0:2^hid_units-1, hid_units)-'0'; % all possible states of <hid_units> hidden units
    P = sum(prod(exp(PS * rbm_w) + 1, 2)); % partition function
    logP = log(P);
end

