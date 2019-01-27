function [diffusion, edgesUsed] = getContinuousDiffusion( A, PropProb, init_inf)
    N = size(A, 1);
    susceptable = ones(1,N);
    unPropagated = sparse(1,N);
    unPropagated( init_inf ) = 1;
    timestamp = -1 * ones(1,N);
    timestamp(init_inf) = 0;
    susceptable(init_inf) = 0;
    edgesUsed = sparse(N,N);
    while (sum(unPropagated) > 0)
        node = find(unPropagated);
        node = node(find( timestamp(node) == min(timestamp(node))));
        node = node(1);
        current_time = timestamp( node );
        unPropagated(node) = 0;
        new_infections = A(node, : ) > rand(1, N );
        new_infections = find(susceptable & new_infections );
        if ( ~isempty(new_infections))
            timeProp = PropProb(1, length(new_infections) );
            timestamp( new_infections ) = current_time + timeProp;
            edgesUsed( node, new_infections) = timeProp;
            unPropagated(new_infections) = 1;
            susceptable( unPropagated > 0 ) = 0;
        end
    end
    diffusion = timestamp;
end