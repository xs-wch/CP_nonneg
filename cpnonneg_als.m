function U = cpnonneg_als(X,R,constrain_dim,update_order,quiet)
% my_cp_als - CP decomposition with ALS
%             some dimension can be constrained to be nonnegtive
%
%
% Syntax:  U = my_cp_als(X,R,constrain_dim)
%
% Inputs:
%    X - tensor, a multiway array
%    R - estimated rank
%    constrain_dim - the dimension index constrained to be nonnegative 
%    update_order - the order to update
%    quiet - true: no display; false: display 
%
% Outputs:
%   U - a structure contains decomposed array
% Example: 
%    tt1 = randn(2,3);
%    tt2 = randn(3,3);
%    tt3 = rand(4,3);
%    test_tensor = zeros(2,3,4);
%    for i = 1:3
%        temp1 = tt1(:,i)*tt2(:,i)';
%        for j = 1:4
%            test_tensor(:,:,j) = test_tensor(:,:,j) + temp1*tt3(j,i);
%        end
%    end
%    U = my_cp_als(test_tensor,3,1);


% Other m-files required: 
% Subfunctions: none
% MAT-files required: tensor toolbox(http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.5.html)
%
% See also: cp_als

% Author: Chaohua Wu
% Department of Biomedical Engineering, Tsinghua University
% email: xs.wuchaohua@outlook.com

% Dec 21 2014;  revision: 

%------------- BEGIN CODE --------------
if nargin < 1
    tt1 = randn(2,3);
    tt2 = randn(3,3);
    tt3 = rand(4,3);
    test_tensor = zeros(2,3,4);
    for i = 1:3
        temp1 = tt1(:,i)*tt2(:,i)';
        for j = 1:4
            test_tensor(:,:,j) = test_tensor(:,:,j) + temp1*tt3(j,i);
        end
    end
    
    X = test_tensor;
    R = 3;
    constrain_dim = 0;
    update_order = [3 1 2];
    quiet = true;
end

dims = size(X);
n_dim = length(dims);
U = cell(n_dim,1);

if (constrain_dim > n_dim) || (constrain_dim < 1)
    constrain_dim = 0;
end
for i = 1:n_dim
    if i ~= constrain_dim
        U{i} = randn(dims(i), R);
    else
        U{i} = rand(dims(i), R);
    end
end

tol = 10^-4;
deltaU = 1;
iter = 0;
while deltaU > tol
    U_old = U;
    for i = update_order(1:n_dim)
        
        
        temp_matrix = my_matricization(X,i);
        [dimseq_without_i,~] = dimseq(n_dim,i);
        dimseq_without_i = dimseq_without_i(end:-1:1);
        P = khatrirao(U{dimseq_without_i});
        if i ~= constrain_dim
            Unew = P\temp_matrix';
            U{i} = Unew';
        else
            %%%%%%%%%%% nonnegtive constrianed least square
            temp_matrix = temp_matrix';
            num_i = size(temp_matrix,2);
            temp_U_i = zeros(size(U{i}))';
            for j = 1:num_i
                temp_U_i(:,j) = lsqnonneg(P,temp_matrix(:,j));
            end
            U{i} = temp_U_i';
            
        end
        
        
        
    end
    deltaU = diffU(U,U_old);
    iter = iter+1;
    if ~quiet
        disp(['iteration number ',num2str(iter),' deltaU = ',num2str(deltaU)]);
    end
end

if quiet
    disp(['iteration number ',num2str(iter),' deltaU = ',num2str(deltaU)]);
end





end


function flatten_matrix = my_matricization(X,md)

dims = size(X);
n_dim = length(dims);
if (md <0) || (md > n_dim)
    error('the dimension to matricization does not exsit');
end
% % if md == 1
% %     flatten_matrix = reshape(X, dims(md),prod(dims(md+1:end)));
% % else if md == n_dim
% %         flatten_matrix = permute(X,[md,1:n_dim-1]);
% %         flatten_matrix = reshape(flatten_matrix, dims(md),prod(dims(1:n_dim-1)));
% %     else
% % 
% %         flatten_matrix = permute(X,[md,1:md-1,md+1:n_dim]);
% %         flatten_matrix = reshape(flatten_matrix, dims(md), prod(dims([1:md-1,md+1:n_dim])));
% %     end
% % end
[dimseq_without_i,dimseq_first_i] = dimseq(n_dim,md);
flatten_matrix = permute(X,dimseq_first_i);
flatten_matrix = reshape(flatten_matrix, dims(md),prod(dims(dimseq_without_i)));
end

function [dimseq_without_i,dimseq_first_i] = dimseq(n_dim,i)

if i == 1
    dimseq_without_i = [i+1:n_dim];
    dimseq_first_i = [1:n_dim];
else if i == n_dim
        dimseq_without_i = [1:i-1];
        dimseq_first_i = [i,1:i-1];
    else

        dimseq_without_i = [1:i-1,i+1:n_dim];
        dimseq_first_i = [i,1:i-1,i+1:n_dim];
    end
end

end

function deltaU = diffU(U,U_old)

n = length(U);
deltaU = 0;
for i = 1:n
    deltaU = norm(U{i}-U_old{i},'fro')/norm(U{i},'fro') + deltaU;
end

end

