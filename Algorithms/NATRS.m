function [title,output,fk,xk]=NATRS(problem,tol)
format long
%********************************************************************
%************* Nonmonotone Adaptive Trust Region Method zhang *************
%**************************************************************************
% step0 : choose the constsnt and variable
title = p00_title ( problem );
n = p00_n ( problem );
xk = p00_start ( problem, n );
v=0.5;
mu=0.1;
N=10;
CON=1;
x_k_2=xk;
k=1; % Number of iteration
fk= p00_f ( problem, n, xk );
gk= p00_g ( problem, n, xk );
g_k_2=gk;
f_eval=1;
flk=fk;
Bk=eye(n);
maxiter=1000;
F=zeros(maxiter,1);
F(k)=fk;
tic
%************************************************************************
% ************* step1 : termination criterion of algorithem *************
while (norm(gk)>=tol && k<=maxiter)
       r=0;
       j=-1;
%*********************************************************************
% ************ step2 : choose qk and definition of sk ****************

       if cond(Bk) < 100
           CON=CON+1;
          qk=0.75*(-inv(Bk)*gk)+0.25*(-gk);
       else
          qk=-gk;
       end
        wk=-(gk'*qk)/(qk'*Bk*qk);
        while (r<mu) % Inner step
               j=j+1;
               delta=v^j*wk*norm(qk);  % Trust Region Radius
%%    step3:solving subproblem of trust region method%%%%%%%%%%%%%
               %dk = Steihaug_Toint(gk,Bk,delta,n,0.1,0,[]);
               dk=More_Sorensen(gk,Bk,delta);
%*****************************************************************
%***** step4 : test of agreement between function and model *****
                % Evaluation of predicted reduction
                pred=-(gk'*dk+0.5*dk'*Bk*dk);
                % Evaluation of actual reduction
                fk= p00_f ( problem, n, xk+dk );

                f_eval=f_eval+1;
                h=flk-fk;
                % test of agreement between function and model
                r=h/pred;
         end % End of inner steps
         xk=xk+dk;
         k=k+1 ;% number of iteration
         x_k_1=x_k_2;
         x_k_2=xk;
%--------------------------------------------------------------------
         mk=min(N,k);
         flk=fk;
         F(k)=fk;
         for i=k-1:-1:k-mk+1
             if F(i)>flk
                flk=F(i);
             end
         end
%--------------------------------------------------------------------
         gk= p00_g ( problem, n, xk );
         g_k_1=g_k_2;
         g_k_2=gk;
%*********************************************************************
% step5: Update of Bk with BFGS formula
         yk=g_k_2-g_k_1;
         dk=x_k_2-x_k_1;
         if (yk'*dk)>0
             Bk=Bk-(Bk*dk*dk'*Bk)/(dk'*Bk*dk)+(yk*yk')/(yk'*dk);
         end
end % end of while , step 1
%*************************************************************************
toc
output=[k f_eval CON];
%************************************************************************
end
% ************************** end of function ****************************