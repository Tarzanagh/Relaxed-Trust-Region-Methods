  function[p,pflag] = nearly_exact(H,g,TR_radius,W)
%
%  Compute the nearly exact solution to the trust region sub problem.
%  We follow Algorithm 4.4 in Nocedal, Wright. 

% First compute the unconstrained minimizer, assuming H is invertible.
R = chol(H);
p = -(R'*R)\g;

% Now check to see if p is within the trust region.
% If it is, take the step.
np = Wnorm(p,W);
if np < TR_radius
    pflag = 0;
    return
else
% If it isn't, approximately solve (see Theorem 4.3, Nocedal and Wright) 
%
%          (H+\lambda*I)p=-g such that ||p||=TR_radius.
%
% by solving 
%
%                      phi(lambda) = 0,        (*)
%
% where phi(lambda) = 1/TR_radius - 1/||(H+lambda*I)\g||. 
% We do this using the implementation of Newton's Method for 
% solving (*) given by Algorithm 4.4.
%
  q = (R')\p;    
  nq = Wnorm(q,W);
  lambda = 0;   
  I = eye(size(H));
  for i = 1:2 
    lambda = lambda + (np/nq)^2*((np-TR_radius)/TR_radius);
    R = chol(H+lambda*I);
    p = -(R'*R)\g;   
    np = Wnorm(p,W);
    q = (R')\p;    
    nq = Wnorm(q,W);
    %fprintf('lambda=%5.5e\n',lambda)
  end
  pflag = 1;
end
function [Wnp] = Wnorm(p,W)

  % Compute the weighted norm ||p||_W, where W is a positive definite
  % diagonal matrix.
  
  Wnp = sqrt(p'*(W.*p));
  
