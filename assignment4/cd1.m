function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    %error('not yet implemented');
    visible_data = sample_bernoulli(visible_data); % for real-valued data, sampling a state to make it binary
    vis_state0 = visible_data;
    hid_prob0 = visible_state_to_hidden_probabilities(rbm_w, vis_state0);
    hid_state0 = sample_bernoulli(hid_prob0);
    G0 = configuration_goodness_gradient(vis_state0, hid_state0);

    vis_prob1 = hidden_state_to_visible_probabilities(rbm_w, hid_state0);
    vis_state1 = sample_bernoulli(vis_prob1);
    hid_prob1 = visible_state_to_hidden_probabilities(rbm_w, vis_state1);
    %hid_state1 = sample_bernoulli(hid_prob1); % before optimization
    hid_state1 = hid_prob1; % no sampling to improve learning rate
    G1 = configuration_goodness_gradient(vis_state1, hid_state1);
    ret = G0 - G1;
end
