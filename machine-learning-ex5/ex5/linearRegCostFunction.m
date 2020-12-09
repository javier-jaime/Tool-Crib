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

h = X*theta;
error = h-y;
error_sqr = sum(error.^2);

theta(1) = 0;
theta_sqr = sum(theta.^2);

J = 1/(2*m)*error_sqr + 1/(2*m)*lambda*(theta_sqr);

grad = (1/m)*X'*error + (1/m)*lambda*theta;


% =========================================================================

grad = grad(:);

end
