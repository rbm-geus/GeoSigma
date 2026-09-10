function reals = get_reals_cholesky(cov,Nreals)

Nvar = size(cov,1);
UT = chol(cov+0.00001*eye(Nvar));

reals = UT'*(randn(Nvar,Nreals));

end