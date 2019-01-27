function [F, G] = obj_cvx(x)
   global densities_mat anti_occurences rho N_act D ndx_node
   %----------------------------------------------------------------------%
   %Generates the objective function and the nonlinear constraints for the
   %convex problem, as well as the 1st order derivatives for the functions.
   %----------------------------------------------------------------------%
    gamma_ndx = 1:D;
    y_ndx = D + (1:N_act);
    
    gamma = x(gamma_ndx);
    y = x(y_ndx);
    F = zeros(D + 1, 1);
    G = zeros(D + 1, length(x));

    F(1) = -sum( gamma ) - sum( anti_occurences(ndx_node) .* y ) + rho * sum( exp(-y) );
    
    G(1,gamma_ndx) = -1 ;
    
    G(1, y_ndx ) = -anti_occurences(ndx_node) - rho * exp(-y);

    exp_y_mat = exp(y) * ones(1, D);
    
    product = prod( 1 - densities_mat(ndx_node,:) + densities_mat(ndx_node,:) .* exp_y_mat );
    
    F( 1 + gamma_ndx ) = log(exp(gamma) + product');
    G( 1 + gamma_ndx, gamma_ndx) = diag(  exp(gamma) ./ ( exp(gamma) + product' )  );
    
    product_mat = product' * ones(1, N_act);
    G( 1 + gamma_ndx, y_ndx) = product_mat .* densities_mat(ndx_node,:)' .* exp_y_mat' ./ (1 - densities_mat(ndx_node,:) + densities_mat(ndx_node,:) .* exp_y_mat)' ./ ( exp(gamma) * ones(1, N_act) + product_mat );
    
end
    