   function [ s, lambda ] = Dennis_Schnabel(g,H,delta,tol)
%------------------------------------------------------------------------
   newton_step = newton_solve(H,g);  % we are redoing this under the
                                     % assumption that it is cheap.
   n_desvar = length(g);

   if ( norm(newton_step,2) < delta )
      s      = newton_step;
      lambda = 0;
   else
      mu = 0.;
      s  = newton_step;
      d_mu = 1.;     % to enter the routine below
      while ( abs(norm(s,2)-delta) > max(tol,sqrt(eps)) )
         r_mu     = norm(s,2) - delta;
         [temp,c] = newton_solve( H + mu*eye(n_desvar), s );
         %  if H is not positive definite, we want to start with
         %  a value of mu so that H+mu*I is.
         mu       = max(mu,c);
         temp     =-temp;
         dr_dmu   =-s'*temp / sqrt( s'*s );
         %  the norm(s)/delta scaling factor results in solving
         %  the secular equation
         d_mu     =-r_mu / dr_dmu     * norm(s,2)/delta;
         mu       = mu + d_mu;
         %  we could reuse the factorization above, however,
	 %  efficiency is not a major concern here.
         s        = newton_solve( H + mu*eye(n_desvar), g );
      end
      lambda      = mu/2;
   end

function [ step, c ] = newton_solve(H,g)
%------------------------------------------------------------------------
%  Solve for the Newton step: step = -H\g.  However, there may be cases
%  when the Hessian is not positive definite.  In this case, we'd like to
%  generate the step: step = -(H + cI)\g, where the non-negative constant
%  c guarantees that (H + cI) is a positive definite matrix.  Doing this,
%  we guarantee that the step is a descent direction, i.e. -g'*step > 0.
   
%  This algorithm comes from Dennis and Schnabel:

   %  "Condition" the matrix H, 
   max_diag = max(diag(H));
   min_diag = min(diag(H));
   if ( min_diag < sqrt(eps)*max(0,max_diag) )
      c        = 2*sqrt(eps)*( max(0,max_diag) - min_diag ) - min_diag;
      max_diag = max_diag + c;
   else
      c        = 0.;
   end

   max_off_diag = max(max( H - diag(diag(H)) ));
   if ( max_off_diag*(1+2*sqrt(eps)) > max_diag )
      c = c + (max_off_diag - max_diag) + 2*sqrt(eps)*max_off_diag;
      max_diag = max_off_diag*( 1+2*sqrt(eps) );
   end

   if ( max_diag == 0 )
      c = 1;
      max_diag = 1; 
   end

   if ( c>0 )
      H = H + c*eye(size(H));
   end   
      
   max_offl = sqrt( max( max_diag, max_off_diag/length(g) ) );

   %  Perform a perturbed Cholesky decomposition on H
   [ L, max_add ] = p_chol( H, max_offl );
   %  If H wasn't positive definite...
   if ( max_add > 0 )
      max_ev = H(1,1);
      min_ev = H(1,1);
      for i=1:length(g),
         off_row = sum(abs(H(i,:))) - abs(H(i,i));
         max_ev = max( max_ev, H(i,i) + off_row );
         min_ev = min( min_ev, H(i,i) - off_row );
      end
      sdd = max( (max_ev-min_ev)*sqrt(eps)-min_ev, 0 );
      c   = min( max_add, sdd );

      H   = H + c*eye(size(H));
      [ L, max_add ] = p_chol( H, 0. );
   end

   c = max(c, max_add);
   %  Now solve the problem (LL')*step = -g;
   lt_step =-L\g;
   step    = L'\lt_step;

function [ L, max_add ] = p_chol( H, max_offl )
%------------------------------------------------------------------------
   n = max(size(H));
   L = zeros(size(H));

   min_l  = sqrt( sqrt(eps) )*max_offl;
   min_l2 = 0.;

   if ( max_offl == 0 )
      max_offl = sqrt( norm(diag(H),inf) );
      min_l2   = sqrt(eps)*max_offl;
   end

   max_add = 0.;

   for j=1:n
      L(j,j) = H(j,j) - L(j,1:j-1)*L(j,1:j-1)';
      min_ljj = 0.;
      for i=j+1:n
         L(i,j) = H(j,i) - L(i,1:j-1)*L(j,1:j-1)';
         min_ljj = max( abs(L(i,j)), min_ljj );
      end
      min_ljj = max( min_ljj/max_offl, min_l );
      if ( L(j,j) > min_ljj^2 )
         %  normal Cholesky iteration...
         L(j,j) = sqrt( L(j,j) );
      else
         %  alter H so the Cholesky iteration can proceed...
         if ( min_ljj < min_l2 )
            %  only possible if max_offl = 0
            min_ljj = min_l2; 
         end
         max_add = max( max_add, min_ljj^2 - L(j,j) );
         L(j,j) = min_ljj;
      end
      L(j+1:n,j) = L(j+1:n,j) / L(j,j);
   end
