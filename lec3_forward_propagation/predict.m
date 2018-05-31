% Forward Propagation for a neural network with one hidden layer
% composed of logistic neurons and the output layer is a linear
% neuron. Bias terms are zeros.
function predictions = predict(W1, W2, X)

    Z=W1'*X';
    
    H=1./(1+exp(-Z));
    O=W2'*H;

    predictions=O';

endfunction

