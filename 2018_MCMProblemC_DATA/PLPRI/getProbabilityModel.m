function [ProbGen, ProbDensity, x_min] = getModel( model_number )

switch (model_number)
    case 1
        %power law
        x_min = 1;
        alpha = 2;
        ProbGen = @(N_imp, M_imp)randPowerLaw(N_imp, M_imp, 1, alpha);
        ProbDensity = @(x)PRDensity(x, x_min, alpha);
    case 2
        %exponential
        x_min = 0;
        alpha = 1;
        ProbGen = @(N_imp, M_imp)(-log(1 - rand(N_imp,M_imp)) / alpha);
        ProbDensity = @(x)(alpha * exp( -alpha * x));
    case 3
        %discrete
        x_min = 0;
        alpha = 0;
        ProbGen = @(N_imp, M_imp)( ones(N_imp, M_imp));
        ProbDensity = @(x)(x == 1);
    case 4
        %weibul
        x_min = 0;
        lambda = 9.479;
        k = 2.3494;
        ProbGen = @(N_imp, M_imp)( lambda*log( 1 ./ (1 - rand(N_imp,M_imp)) ).^(1/k) );
        ProbDensity = @(x)( (k / lambda) * (x / lambda).^(k-1) .* exp( -(x/lambda).^k) );
    otherwise
        fprintf('ERROR: Invalid Incubation model number!\n')
        return
end

end

function probs = PRDensity( x, x_min, alpha )
probs = (alpha - 1) / x_min * (x / x_min) .^ (-alpha);
probs( find( x < x_min) ) = 0;
end

function output = randPowerLaw( N, M, x_min, alpha )
y = rand(N, M);
output = (x_min .^ (1 - alpha) - y .* x_min .^(1 - alpha)).^(1/(1-alpha));
end
