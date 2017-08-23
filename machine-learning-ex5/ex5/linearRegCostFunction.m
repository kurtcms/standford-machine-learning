function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

p = X * theta;
sqe = (p - y)' * (p - y) ;
J = sum(((1/(2 * m)) * sqe) + ((lambda/(2 * m)) * (theta' * theta - theta(1)^2)));

for i = 1:length(theta);
    if i == 1;
        grad(i) = 1/m * ((p - y)' * X(:, i));
    else;
        grad(i) = (1/m * ((p - y)' * X(:, i))) + lambda/m * theta(i);
    endif;
end

% =========================================================================

grad = grad(:);

end
