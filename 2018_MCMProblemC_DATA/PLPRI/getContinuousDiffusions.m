function [diffusions, infectionTraces] = getDiffusions( A, PropProb)

N = size(A,1);
diffusions = zeros( 2 * length(A), length(A) );
edgesUsed = zeros(N,N);
count = 1;
numTrueEdges = sum(sum(A > 0.01));
infectionTraces = {};

while ( sum(sum(edgesUsed > 0)) <= .99 * numTrueEdges)
   [ diffusion, eu ] = getContinuousDiffusion( A, PropProb, ceil( N * rand())  );
   while ( length(find(diffusion > 0)) == 0)
        [ diffusion, eu ] = getContinuousDiffusion( A, PropProb, ceil( N * rand()) );
   end
   %[sum(sum(edgesUsed > 0)) / numTrueEdges count]
   edgesUsed = edgesUsed + (eu > 0);
   if (count > size(diffusions,1))
       diffusions = [ diffusions' zeros(length(A), 2 * length(A) )]';
       infectionTraces{size(diffusions,1)} = zeros(N,N);
   end
   diffusions(count,:) = diffusion;
   infectionTraces{count} = eu;
   count = count + 1;
end
diffusions = diffusions( 1:(count-1), :);

end
