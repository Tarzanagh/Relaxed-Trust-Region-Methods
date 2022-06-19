  function [s,cg_iter,term_code,snorm,m_reduction] =...
           Steihaug_Toint(g,H,D,max_cg_iter,resid_tol_factor,io_flag,cg_fig )
 %function [s,cg_iter,term_code,snorm,m_reduction] = cg_1(x,g,cost_params,opt_params);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Use preconditioned CG iteration to approximately solve the trust region 
%  subproblem
%       min s'*g + .5*s'*H*s   subject to   ||s||_M <= D
%  using the Steihaug-Toint truncated CG method.
%
%  Reference: Algorithm 7.5.1 on p. 205 of A.R. Conn, N.I.M. Gould,
%  and P.L. Toint's book "Trust-Region Methods", SIAM, 2000.
%
%  Termination Code           Stopping Criterion
%    term_code = 1    Negative curvature (negative eigs of H) detected.
%    term_code = 2    CG iterate s outside trust region.
%    term_code = 3    Maximum CG iterations reached.
%    term_code = 4    CG stopping tolerance reached.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% D           = opt_params.tr_radius;
%  max_cg_iter = opt_params.max_cg;
%  resid_tol_factor = opt_params.resid_tol_factor;%0.1
%  io_flag     = opt_params.cg_io_flag; % If >= 1, output CG stopping info
%  cg_fig      = opt_params.cg_figure_no;
%  M_inverse   = cost_params.M_inverse_fn;
%  M_inner_product = cost_params.M_inner_product;
%  Hmult       = cost_params.Hess_mult_fn;
%  cost_fn     = cost_params.cost_fn;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Compute and save terms used in all CG iterations.       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% cost_params.save = 1;
%  [J,cost_params,g]=feval(cost_fn,x,cost_params);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  CG initialization.                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  s = vec_scalar_mult(0,g);               %  Initialize s = zeros(size(g)).
  r = g;                                  %  r = H*s + g evaluated at s = 0.
 % if isempty(M_inverse)
  %  y = r;  %  No preconditioning.
 % else
 %   y = feval(M_inverse,r,cost_params);   %  y = M^{-1}*r.
 % end
  y = r;
  p = vec_scalar_mult(-1,y);              %  p = -y.
  delta = vec_dot_product(r,y);           %  delta = r'*M^{-1}*r.
  term_code = 0;
  cg_iter = 0;
  n_eval = 0;
  residnormvec = [];
  stepnormvec = [];
  
  %  Compute CG residual stopping tolerance.

  gnorm = sqrt(delta);
  resid_tol = min(resid_tol_factor,sqrt(gnorm))*gnorm;

  %  Terms used to recursively compute ||s||_M and quadratic model reduction.

  s_normsq = 0;
  p_normsq = delta;
  sp_innerprod = 0;
  m_reduction = 0;
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  CG iteration.                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  while term_code == 0
    cg_iter = cg_iter + 1;
    residnormvec = [residnormvec; delta];
    
    %  Plot CG convergence information
    
    if ~isempty(cg_fig)
      figure(cg_fig)
      lngth = max(size(residnormvec))-1;
      subplot(221)
        semilogy([0:lngth],residnormvec,'o')
 	title('Squared Residual M^{-1}-Norm')
	xlabel('CG iterate')
      if ~isempty(stepnormvec)
	indx = [1:lngth]';
	subplot(222)
        semilogy(indx,stepnormvec,'o', indx,D*ones(lngth,1))
	title('Squared M-Norm of CG Step')
	xlabel('CG iterate')
      end
      drawnow
    end
    
    %Hp = feval(Hmult,p,x,cost_params);
    Hp = H*p;
    pHp = vec_dot_product(p,Hp);
    
    if pHp <= 0
      term_code = 1; 			%  Negative curvature detected.
      if io_flag
        fprintf('   --- Negative curvature detected in CG.\n');
      end
      [s,tau] = bndry_solve(s,p,D,s_normsq,sp_innerprod,p_normsq);
      snorm = D;
      m_reduction = m_reduction - tau*delta + 0.5*tau^2*pHp;
      return
    end %(end negative curvature)

    alpha = delta / pHp;
    s_new = vec_axpy(alpha,p,s);  %  s_new := s + alpha*p
    s_new_normsq = s_normsq + 2*alpha*sp_innerprod + alpha^2*p_normsq;
    
    if sqrt(s_new_normsq) >= D       % CG step outside the trust region.
      term_code = 2;   
      if io_flag 
        fprintf('   --- CG iterate outside of trust region.\n');
      end 
      % Take step to boundary.
      [s,tau] = bndry_solve(s,p,D,s_normsq,sp_innerprod,p_normsq);
      snorm = D;
      m_reduction = m_reduction - tau*delta + 0.5*tau^2*pHp;
      return
    end
    %  Update CG step, 
    %    s := snew = s + alpha*p.
    
    s = s_new;
    s_normsq = s_new_normsq;
    snorm = sqrt(s_normsq);
    m_reduction = m_reduction - 0.5*alpha*delta;
    stepnormvec = [stepnormvec; snorm];

    if cg_iter >= max_cg_iter
      term_code = 3;
      if io_flag
        fprintf('   --- Max CG iterations exceeded.\n');
      end  
      return
    end
    
  %  Update CG residual,
  %      r := r + alpha * H*p
  
    r = vec_axpy(1,r,vec_scalar_mult(alpha,Hp)); 

    %  Apply preconditioner,
    %    y := M^{-1}*r.

   % if isempty(M_inverse)
    %  y = r;  %  No preconditioning.
    %else
    %  y = feval(M_inverse,r,cost_params);
    %end
    y = r;    
    %  Computed weighted residual squared norm, 
    %    delta = r'*M^{-1}*r.
    
    delta_new = vec_dot_product(r,y);
    
    if sqrt(delta_new) <= resid_tol
      term_code = 4;
      if io_flag
        fprintf('   --- Residual stopping tolerance reached in CG.\n');
      end  
      return
    end

   %  Update CG search direction, 
   %    p := -y + beta*p.
   
    beta = delta_new / delta;
    p = vec_axpy(-1,y,vec_scalar_mult(beta,p));
    delta = delta_new;

    %  Terms for ||s||_M recursion.

    sp_innerprod = beta*(sp_innerprod + alpha*p_normsq);
    p_normsq = delta + beta^2*p_normsq;
    
  end %(while term_code == 0)
    function z = vec_scalar_mult(a,x);

%  If x is a cell array, compute the new cell array
%      z{i} = a*x{i}.
%  Otherwise, compute the usual scalar multiple z = a*x.

  if sum(size(a)) ~= 2
    fprintf('*** 1st argument must be a scalar in VEC_AXPY.M.\n');
    return
  end
  if iscell(x)
    n_cellx = max(size(x));
    for i = 1:n_cellx
      z{i} = a*x{i};
    end

  else
    z = a*x;
  end
    function c = vec_norm(x)

%  If x is a cell array, compute the square root of the sum or the squared
%  norms of the vectors consisting of the components x, i.e.,
%      c = sum_i ||x{i}||^2.
%  Otherwise, compute the norm of the vector x.

  if iscell(x)
    n_cells = max(size(x));
    c = 0;
    for i = 1:n_cells
      xi = x{i};
      c = c + norm(xi(:))^2;
    end
    c = sqrt(c);

  else
    c = norm(x(:));
  end
    function c = vec_dot_product(x,y);

%  If x and y are cell arrays, compute the sum of dot products of the 
%  component vectors, i.e.,
%      c = sum_i dot_product(x{i},y{i}).
%  Otherwise, compute the usual dot the vectors x and y.

  if iscell(x)
    n_cellx = max(size(x));
    n_celly = max(size(y));
    if n_cellx ~= n_celly
 fprintf('*** Cell arrays x, y in VEC_DOT_PRODUCT.M must be the same size.\n');
      return
    end
    c = 0;
    for i = 1:n_cellx
      xi = x{i};
      yi = y{i};
      c = c + xi(:)'*yi(:);
    end

  else
    c = x(:)'*y(:);
  end
    function z = vec_axpy(a,x,y);

%  If x and y are cell arrays, compute the new cell array
%      z{i} = a*x{i} + y{i}.
%  Otherwise, compute the usual z = a*x+y.

  if sum(size(a)) ~= 2
    fprintf('*** 1st argument must be a scalar in VEC_AXPY.M.\n');
    return
  end
  if iscell(x)
    n_cellx = max(size(x));
    n_celly = max(size(y));
    if n_cellx ~= n_celly
      fprintf('*** Cell arrays x, y in VEC_AXPY.M must be the same size.\n');
      return
    end
    for i = 1:n_cellx
      xi = x{i};
      yi = y{i};
      z{i} = a*xi + yi;
    end

  else
    z = a*x + y;
  end



     function [s,tau] = bndry_solve(s,p,Delta,s_normsq,sp_innerprod,p_normsq)

%  Find the positive root tau of the quadratic equation
%     ||s + tau*p||_M = Delta,
%  where ||.||_M denotes the M-inner product
%     ||v||_M = sqrt(v'*M*v).
%  Then update
%     s := s + tau*p.

  a = p_normsq;
  b = 2 * sp_innerprod;
  c = s_normsq - Delta^2;
  tau = (-b + sqrt(b^2 - 4*a*c))/(2*a);
  s = vec_axpy(tau,p,s);
