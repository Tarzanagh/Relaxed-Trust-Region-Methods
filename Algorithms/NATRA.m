function [title,output,fk,sol]=NATRA(problem,tol,dim)
%nonmonotone trust-region line search method forlarge-scale unconstrained
%optimization
%********************************************************************
%************* Nonmonotone Adaptive Trust Region Method by shi:qk=-inv(A)*gk; *************
%**************      Test Problem by More            *************
%**************nonmonotone term :flk *************
%%                    step1: input 
format long
mu_1=0.1;
mu_2=0.9;
etak2=0.15;
etak=(etak2)/2;
delta_max=125;
title = p00_title ( problem );
n = dim;
%n=100;
xk = p00_start ( problem, n );
N=5;
xk2=xk;
k=1;
fk= p00_f ( problem, n, xk );
gk= p00_g ( problem, n, xk );
gk2=gk;
FE=1;
Rk=fk;
v=1;
vk=1;
A=eye(n);
maxit=50000;
maxnf=50000;
F=zeros(maxit,1);
F(k)=fk;
tic 
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
          output = [total_iters,FE,fk];
          fprintf('%s%d', ' Successful Termination , ierr = ',ierr )
          toc
            return;
      end
 %%   step2:  choose qk and definition of sk %%%%%%%%%%%%% 
       qk=-gk;
       wk=-(gk'*qk)/(qk'*A*qk);        
%%    step3:solving subproblem of trust region method%%%%%%%%%%%%%
      while 1,
            vk=v*vk;
            delta=min(vk*wk*norm(qk),delta_max);
            dk = Steihaug_Toint(gk,A,delta,n,0.1,0,[]);
           % dk = More_Sorensen(gk,A,delta);
%%           step 4: test of agreement between function and model %%%%%%
             pred=-(gk'*dk+0.5*dk'*A*dk);
             fk= p00_f ( problem, n, xk+dk );
             FE=FE+1;
             ared=Rk-fk;
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
      %%
      mk=min(N,k);
      flk=fk;
      F(k)=fk;
      for i=k-1:-1:k-mk+1
          if F(i)>flk
            flk=F(i);
         end
      end
      etak1=etak2;
      etak2=etak;
      etak=(etak1+etak2)/2;
      Rk=etak*flk+(1-etak)*fk;
      %%
      gk= p00_g ( problem, n, xk );
      gk1=gk2;
      gk2=gk;     
      y_k=gk2-gk1;
      x_k=xk2-xk1;
      if (y_k'*x_k)>0   
         A=A-(A*x_k*x_k'*A)/(x_k'*A*x_k)+(y_k*y_k')/(y_k'*x_k);
      end
end 
end