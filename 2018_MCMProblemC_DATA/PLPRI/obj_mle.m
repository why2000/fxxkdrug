function [f, gradient] = obj_mle(x)
   global densities_mat relevant_cascades anti_occurences rho
   %----------------------------------------------------------------------%
   %Generates the objective function  non-convex likelihood problem, as
   %well as the 1st order derivatives.
   %----------------------------------------------------------------------%
   N = length(x);
   gradient = zeros(N,1);

   x_mat = x * ones(1,length(relevant_cascades));

   difference_mat = 1 - x_mat .* densities_mat;
   product = prod( difference_mat );
   others = -ones(N,1) * product ./ difference_mat .* densities_mat ./ ( ones(N,1) * (1 - product) ) ;
   f = - sum( log( 1 - product) );
   gradient = gradient + sum(others')';
   f = f - sum( anti_occurences .* log( 1 - x ) );
   gradient = gradient + anti_occurences ./ ( 1 - x );
   f = f + rho * sum( abs(x) );
   gradient = gradient';
   gradient = gradient + rho;
end