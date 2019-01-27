function [A_mle, precision, recall, mse] = connie( varargin )
global densities_mat relevant_cascades anti_occurences rho N_act D node_ndx ndx_node

%-------------------------------------------------------------------------%
%
%Implementation of the ConNIe (CONvex Network InferencE)  algorithm. (see
%   http://snap.stanford.edu/connie)
%
%Infers a hidden network (the adjacency matrix) connecting a set of nodes,
%   given a set of cascades i.e. node infection times that have propagated 
%   through the network.
%
%[A_mle] = connie( rho, incubation_model_number, diffusions, suboptimal_tol )
%   Infers the adjacency matrix from the given set of cascades
%   diffusions.  The ith column of the jth row of diffusion is the 
%   time at which node i was infected by cascade j.  If this value is -1, 
%   then node i was never infected by cascade j.  The parameter rho 
%   is the coefficient on the sparsity penalty function.  The larger
%   rho is, the more sparse A_mle will be.  In other words, increasing 
%   rho will increase inferred edge precision but will decrease recall.
%
%   The arguement suboptimal_tol is the fraction of negative lagrange
%   multipliers an accepted network can have.  Because SNOPT is a general
%   nonlinear solver that does not exploit convexity, it is actually
%   quicker to solve the non-convex MLE problem instead of the convex
%   problem.  Once a solution has been found in the non-convex problem, it
%   is plugged into the KKT conditions of the convex problem.  If the
%   fraction of lagrange multipliers that are negative is less than
%   suboptimal_tol, then the solution is accepted.  Otherwise, the problem
%   is re-solved using the convex problem.  If suboptimal_tol = 1, then the
%   convex problem is never used.  If suboptimal_tol = 0, then A_mle will
%   be the true globally optimal inferred network.
%
%   The incubation_model_number specifies which infection incubation
%   density function to assume.  Below specifies which value of
%   incubation_model_number corresponds to which density function.
%
%   1 - Power law -> w(t) ~ t^(-2) (typical of information propagation)
%   2 - Exponential -> w(t) ~ exp(-t)
%   3 - Discrete -> Incubation times of every infection is exactly t=1
%   4 - Weibul -> w(t) ~ t / lambda).^(k-1) .* exp( -(t/lambda).^k) )
%        where lambda = 9.479 and k = 2.3494. (params correspond to SARS
%        outbreak in Hong Kong.
%
%
%[A_mle, precision, recall, mse] = connie( rho, incubation_model_number, diffusions, A_true, suboptimal_tol )
%   A_true is the ground truth of the hidden network.  The precision and
%   recall of edge detection will be calculated, as well as mean square
%   error (MSE) of the edge weights (infection probabilities).
%
%[A_mle, precision, recall, mse] = connie( A_true, rho, incubation_model_number, suboptimal_tol )
%   A_true is the ground truth of the hidden network.  Cascades are
%   synthetically generated using the specified incubation model
%
%-------------------------------------------------------------------------%
if ( nargin == 4 )
    if ( length( varargin{1} ) == 1)
        A_true = [];
        rho = varargin{1};
        diffusions = varargin{3};
        incubation_model_number = varargin{2};
        N = size(diffusions,2);
    else
       A_true = varargin{1};
       N = length(A_true);
       diffusions = [];
       rho = varargin{2};
       incubation_model_number = varargin{3};
    end
    
elseif ( nargin == 5 )
        A_true = varargin{4};
        N = length(A_true);
        rho = varargin{1};
        diffusions = varargin{3};
        incubation_model_number = varargin{2};    
else
    fprintf('Error - Invalid number of arguments provided!\n')
    return
end
suboptimal_tol = varargin{end};
if ( ~issparse(A_true) )
    A_true = sparse(A_true);
end
A_true = A_true - diag(diag(A_true));

minEdge = 1e-2;
maxEdge = .99;

[ProbGen, ProbDensity, x_min] = getProbabilityModel( incubation_model_number );

if (size(diffusions,1) == 0)
    fprintf('Generating diffusions...\n')
    diffusions = getContinuousDiffusions( A_true, ProbGen);
    fprintf('Done!  %.1d Diffusions Generated\n', size(diffusions,1) );
end

A_mle = sparse(N,N);

fprintf('-----------------------------------------------------------------------------------------------\n');
fprintf('| Node |\t Precision \t| \tRecall\t | \tMSE\t | Runtime | Frac. Suboptimal |\n');
fprintf('-----------------------------------------------------------------------------------------------\n');
for node=(1:N)
    tic;
    
    %Preprocessing - This dramatically speeds up the objective function
        %evaluations.
    anti_occurences = zeros(N,1);
    relevant_cascades = zeros(1,size(diffusions,1));
    transObs = zeros(N,1);
    for d=(1:size(diffusions,1))
        diffusion = diffusions(d,:);
        time = diffusions(d,node);
        others = find((diffusion + x_min) < time & diffusion > -1);
        if (diffusions(d,node) > 0)
            relevant_cascades(d) = 1;
            transObs(others) = 1;
        else
            for i=(1:N)
                anti_occurences(i) = anti_occurences(i) + ( diffusions(d,i) >= 0 );
            end
        end
    end
    ndx_node = find(transObs);
    node_ndx = zeros(N,1);
    for i=(1:length(ndx_node))
       node_ndx( ndx_node(i) ) = i; 
    end
    anti_occurences(node) = 0;
    relevant_cascades = find( relevant_cascades > 0);
    densities_mat = zeros( N, length(relevant_cascades));
    for d_ndx=(1:length(relevant_cascades))
        d = relevant_cascades(d_ndx);
        time = diffusions(d, node);
        diffusion = diffusions(d,:);
        infected = find( (diffusion + x_min) < time & diffusion > -1 );
        density = ProbDensity( time - diffusions(d,infected) );
        densities_mat( infected, d_ndx ) =  density;
    end

    if ( node == 1)
        x0 = ones(N,1) / N;
    else
        x0 = mean( A_mle(:, (1:node))' )';
    end
      
    n = length(x0);
    if ( n > 0 & ~isempty(x0) & length(relevant_cascades) > 0)
        %solving the nonconvex problem to find A_mle's sparsity pattern

        xlow = (minEdge / 5) * ones(n,1);
        xupp = maxEdge * ones(n,1);
        xmul = zeros(n,1);
        xstate = zeros(n,1);
        Flow = -Inf;
        Fupp = Inf;
        Fmul = 0;
        Fstate = 0;
        ObjAdd = 0;
        ObjRow = 1;
        iAfun = ones(n,1);
        jAvar = (1:n)';
        A = 0 * ones(n,1);
        iGfun = ones(n,1);
        jGvar = (1:n)';

        D = length(relevant_cascades);
        [x,F,xmul,Fmul,INFO]= snsolve( x0, xlow, xupp, xmul, xstate,    ...
                           Flow, Fupp, Fmul, Fstate,       ...
                           ObjAdd, ObjRow, A, iAfun, jAvar,...
                           iGfun, jGvar, 'obj_mle');


        snprint off;
        
        
        if (rho > 0)
            %now that the edge locations have been deterermined, optimize edge weights 
            rho_temp = rho;
            rho = 0;
            zeroed = find( x < minEdge);
            nonzeroed = find( x >= minEdge);
            %xlow(nonzeroed) = minEdge;
            xupp(zeroed) = minEdge / 5;
            x0 = x;
            x0(zeroed) = 0;

            [x,F,xmul,Fmul,INFO]= snsolve( x0, xlow, xupp, xmul, xstate,    ...
                               Flow, Fupp, Fmul, Fstate,       ...
                               ObjAdd, ObjRow, A, iAfun, jAvar,...
                               iGfun, jGvar, 'obj_mle');
            x( zeroed ) = 0;
            N_act = length(ndx_node);

            gamma = zeros(D,1);
            for d=(1:D)
               neighbors = find( densities_mat(:,d) > 0.0);
               gamma(d) = 1 - prod( 1 - densities_mat(neighbors,d) .* x(neighbors) ); 
            end
            
            %check to see how close the solution is to optimal
            y = 1 - x(ndx_node);
            x0_cvx = [ gamma' y']';
            [F_cvx, G_cvx] = obj_cvx(x0_cvx);
            delF_0 = G_cvx(1,:)';
            tight_c = find( F_cvx(2:end) >= 0 );
            tight_x_below = find( x0_cvx >= 0 );
            tight_x_above = intersect( find( x0_cvx < 0), zeroed );
            delFC = G_cvx(2:end,:)';
            delFC = delFC(:, tight_c);
            delX_below = eye( length(x0_cvx) );
            delX_below = delX_below(:, tight_x_below);
            delX_above = -eye( length(x0_cvx) );
            delX_above = delX_above(:, tight_x_above);
            delFC = [delFC delX_below delX_above];
            lambda = delFC \ -delF_0;

            frac_suboptimal =  length(find(lambda < 0)) / length(lambda) ;
            
            %If it is outside the suboptimal tolerance, solve the convex
            %   problem
            if  ( frac_suboptimal > suboptimal_tol )
                ndx_node = setdiff(ndx_node, zeroed);
                N_act = length(ndx_node);
                n_cvx = D + N_act;

                y = 1 - x(ndx_node);
                x0_cvx = [ gamma' y']';
                xlow_cvx = -Inf * ones(n_cvx,1);
                xupp_cvx = zeros(n_cvx,1);
                Flow_cvx = -Inf * ones(D+1,1);
                Fupp_cvx = 0 * ones(D+1,1);
                Fupp_cvx(1) = Inf;
                Flow_cvx(1) = -Inf;
                Fstate_cvx = 0 * ones(D+1,1);

                iGfun_cvx = zeros( (D+1)*n_cvx, 1);
                jGvar_cvx = zeros( (D+1)*n_cvx, 1);

                iGfun_cvx( (1:n_cvx) ) = ones(n_cvx,1);
                jGvar_cvx( (1:n_cvx) ) = 1:n_cvx;

                count = N_act+1;
                for d = (1:(D))
                   iGfun_cvx( count ) = d+1;
                   jGvar_cvx( count ) = d;
                   neighbors = find( densities_mat(:,d) > 0.0 );
                   iGfun_cvx( count + (1:length(neighbors)) ) = d;
                   jGvar_cvx( count + (1:length(neighbors)) ) = D + neighbors;
                   count = count + length(neighbors) + 1;
                end

                snseti('Superbasics limit', n_cvx);
                [x_cvx,F_cvx,INFO_cvx, xmul_cvx,Fmul_cvx] = snopt(x0_cvx,xlow_cvx,xupp_cvx,Flow_cvx,Fupp_cvx,'obj_cvx' );
                if ( INFO_cvx == 1)
                    x(ndx_node) = 1 - exp(x_cvx(D+1:end));
                end
            end

            
            %----------------------------
            
            
            rho = rho_temp;
        end
        x(node) = 0;

        A_mle(:,node) = x;
        fprintf('%.1d%\t', node);
        if ( length(A_true) > 0)
            x_real = A_true(:,node);
            edges_predict = find(abs(x) > minEdge);
            edges_real = find( abs(x_real) >= minEdge);
            common = intersect(edges_real, edges_predict);
            all_edge = union( edges_real, edges_predict);
            mse = norm(x(all_edge) - x_real(all_edge), 'fro')^2 / length(all_edge);
            precision = length(common) / length(edges_predict);
            recall = length(common) / length(edges_real);
            fprintf('\t%17.4f\t%15.4f\t%12.4f', precision, recall, mse);
        else
            fprintf('\t\t\t?\t\t?\t\t?')
        end
        rt = toc;
        fprintf('\t%10.4f\t%8.4f', rt, frac_suboptimal);
        if  ( frac_suboptimal > suboptimal_tol )
            fprintf('*');
        end
        fprintf('\n')

        %fprintf(1, 'snopt runtime: %f\n', toc)
    end
end

fprintf('* convex problem was solved\n')

if ( length(A_true) > 0)
    edges_predict = find(abs(A_mle) > minEdge);
    edges_real = find( abs(A_true) >= minEdge);

    common = intersect(edges_real, edges_predict);
    all_edge = union( edges_real, edges_predict);

    mse = norm(A_mle(all_edge) - A_true(all_edge), 'fro')^2 / length(all_edge);

    precision = length(common) / length(edges_predict);

    recall = length(common) / length(edges_real);

    fprintf('\n-------\n');
    fprintf('|Total|\n')
    fprintf('-------\n');
    fprintf( 'Precision: %f \nRecall: %f \nMean Square Error %f \n', precision, recall, mse)
end

end
											

