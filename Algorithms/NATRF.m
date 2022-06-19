function [title,output,sol,fk,e]=NATRF(problem,tol)

%nonmonotone trust-region line search method forlarge-scale unconstrained
%optimization
%Ataee Tarzanagh, D., Peyghami, M. Reza, Bastin, F.: A new nonmonotone adaptive retrospective trust region method for unconstrained optimization problems, Journal of Optimization Theory and Applications, 2015, DOI 10.1007/s10957-015-0790-0
%Peyghami, M. Reza, Ataee Tarzanagh, D.: A relaxed nonmonotone adaptive trust region method for solving unconstrained optimization problems, Computational Optimization and Applications, 61(2)(2015):321-341.
%Ataee Tarzanagh, D., Z. Saeidian, Peyghami, M. Reza, Mesgharani, H.: A new trust region method for solving least-square transformation of system of equalities and inequalities, Optimization Letters, 9(4)(2015):283-310.
%Ataee Tarzanagh, D., Peyghami, M. Reza, Mesgharani, H.: A new nonmonotone trust region method for unconstrained optimization equipped by an efficient adaptive radius, Optimization Methods & Software, 29(4) (2014):819–836.


%********************************************************************
%************* Nonmonotone Adaptive Trust Region Method by shi:qk=-inv(A)*gk; *************
%**************      Test Problem by More            *************
%**************nonmonotone term :flk *************
%%                    step1: input 
format long
mu_1=0.1;
mu_2=0.9;
title = p00_title ( problem );
n = p00_n ( problem );
xk = p00_start ( problem, n );
CON=1;
N=10;
xk2=xk;
k=1;
fk= p00_f ( problem, n, xk );
gk= p00_g ( problem, n, xk );
gk2=gk;
FE=1;
flk=fk;
v=1;
vk=1;
A=eye(n);
maxit=5000;
maxnf=5000;
F=zeros(maxit,1);
F(k)=fk;
T = cputime; 
%%           step2 : termination  condition   
while 1,
    if(k >= maxit)
        ierr = 1;
        sol = xk;
        total_iters = k;
        output = [total_iters,FE];
        fprintf('% s % d','FAILURE, ierr= ',ierr);
        return;
     end
     if(FE >= maxnf)
         ierr = 2;
         sol = xk;
         total_iters = k;
         output=[total_iters,FE];
         fprintf('% s % d','FAILURE,  ierr= ',ierr)   
         return;
      end
      if (norm(gk) <=  tol)
          ierr = 0;
          sol = xk;
          total_iters = k;
          output = [total_iters,FE];
          fprintf('%s%d', ' Successful Termination , ierr = ',ierr )
          e=cputime-T;
            return;
      end
 %%   step2:  choose qk and definition of sk %%%%%%%%%%%%% 
       if cond(A) < 100 
           CON=CON+1;
          qk=0.75*(-inv(A)*gk)+0.25*(-gk);
       else
          qk=-gk;
       end
      wk=-(gk'*qk)/(qk'*A*qk);        
%%    step3:solving subproblem of trust region method%%%%%%%%%%%%%
      while 1,
            vk=v*vk;
            delta=vk*wk*norm(qk);
            [dk]=More_Sorensen(gk,A,delta);
%%           step 4: test of agreement between function and model %%%%%%
             pred=-(gk'*dk+0.5*dk'*A*dk);
             fk=p00_f(problem,n,xk+dk); 
             FE=FE+1;
             ared=flk-fk;
             r=ared/pred;
             if(r>=mu_2)
                   v=5;
                   break;
             elseif(mu_1<=r) &&(r<mu_2)           
                   v=1;
                   break;
             else
                   v=1/5;
             end
      end
%%    step5 : Update of A ,constsnt and variable %%%%%%%%%%%%%%       
      xk=xk+dk;
      k=k+1;  
      xk1=xk2;
      xk2=xk;
      mk=min(N,k);
      flk=fk;
      F(k)=fk;
      for i=k-1:-1:k-mk+1
          if F(i)>flk
            flk=F(i);
         end
      end
      gk=p00_g(problem,n,xk);
      gk1=gk2;
      gk2=gk;     
      y_k=gk2-gk1;
      x_k=xk2-xk1;
      if (y_k'*x_k)>0   
         A=A-(A*x_k*x_k'*A)/(x_k'*A*x_k)+(y_k*y_k')/(y_k'*x_k);
      end
end 
end