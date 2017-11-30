function [Df,Db,errP] = trainCoupledD(Df, Db, Xf, Xb, p)

iternum = p.itern;
%number of patches
nItems = size(Xf,2);
ds = p.ds;
ep = p.ep;
DUC = p.DUC
Df = Df*diag(1./sqrt(sum(Df.*Df)));
Db = Db*diag(1./sqrt(sum(Db.*Db)));

rItem = zeros(1,ds);
uI = 1:nItems;


%   main loop through iterations
for iDUC = 1: DUC
    for iter = 1:iternum
        iter
        Gamma = jointgradSparse(ep,Xf,Df,Xb,Db,iter);
        %     tst = zeros(1,ds);
        %     for i1 = 1:ds
        %         tst(i1) = nnz(Gamma(i1,:));
        %     end
        %     if find(tst==0)
        %         disp(['Unused atoms in coupled Dictionary' num2str(find(tst==0)) ])
        %     end
        %
        %   dictionary update
        if mod(iter,2)>0
            d = 1 : ds;
            for j = 1:ds
                [Db(:,d(j)),Df(:,d(j)),gMask_jb,gMask_jf,dIndice,uI,rItem] = optimize_atom(Xb,Xf,Db,Df,d(j),Gamma,uI,rItem);
                Gamma(d(j),dIndice) = gMask_jf;
            end
        else
            d = 1:ds; % randperm(ds);  %
            for j = 1:ds
                [Df(:,d(j)),Db(:,d(j)),gMask_jf, gMask_jb,dIndice,uI,rItem] = optimize_atom(Xf,Xb,Df,Db,d(j),Gamma,uI,rItem);
                Gamma(d(j),dIndice) = gMask_jb;
            end
            G1=Db'*Db;
            G2=Df'*Df;
            diff=0;
            for ii=1:nItems
                b1=-Db'*Xb(:,ii);
                Gamma_test1= gradSparse(ep, G1, b1);
                b2=-Df'*Xf(:,ii);
                Gamma_test2= gradSparse(ep,G2, b2);
                diff = diff + norm(Gamma_test1-Gamma_test2);
            end
            errP(iter/2)= diff
        end
        %clear unused atoms
        [Df,Db,deItem] = cleardict(Df,Db,Gamma,Xf,Xb,uI,rItem);
    end
end
return
end


function [atom1,atom2,gamma_j1,gamma_j2,dIndice,uI,rItem] = optimize_atom(X1,X2,D1,D2,j,Gamma,uI,rItem)

% data samples which use the atom, and the corresponding nonzero
% coefficients in Gamma
dIndice = find(Gamma(j,:));

if (length(dIndice) < 1)
    maxsignals = 60000;
    perm = randperm(length(uI));   %   1:length(uI);   %
    perm = perm(1:min(maxsignals,end));
    Err = X1(:,uI(perm)) - D1*Gamma(:,uI(perm));
    E = sum(Err.^2);
    [~,i] = max(E);
    atom1 =X1(:,uI(perm(i)));
    atom1 = atom1./norm(atom1);
    atom1 = sign(atom1(1)) * atom1;
    atom2 =X2(:,uI(perm(i)));
    atom2 = atom2./norm(atom2);
    atom2 = sign(atom2(1)) * atom2;
    gamma_j= zeros(length(dIndice),1);
    uI = uI([1:perm(i)-1,perm(i)+1:end]);
    rItem(j) = 1;
    return;
end

gamma_j = Gamma(j,dIndice);
smallGamma = Gamma(:,dIndice);
Dj1 = D1(:,j);
Dj2 = D2(:,j);
[atom1,s,gamma_j1] = svds(X1(:,dIndice) - D1*smallGamma + Dj1*gamma_j, 1);
[atom2,s,gamma_j2] = svds(X2(:,dIndice) - D2*smallGamma + Dj2*gamma_j, 1);

end

function [Df,Db,deItem] = cleardict(Df,Db,Gamma,Xf,Xb,uI,rItem)

maxTh = 4;
th = 0.99;
ds = size(Df,2);

% compute error in pa to conserve memory
diff = zeros(1,size(Xf,2));
pa = [1:3000:size(Xf,2) size(Xf,2)+1];
for i = 1:length(pa)-1
    error_big(:,pa(i):pa(i+1)-1) = Xf(:,pa(i):pa(i+1)-1)-Df*Gamma(:,pa(i):pa(i+1)-1);
    err_big(pa(i):pa(i+1)-1) = sum(error_big(:,pa(i):pa(i+1)-1).^2);
end
diff = zeros(1,size(Xb,2));
pa = [1:3000:size(Xb,2) size(Xb,2)+1];
for i = 1:length(pa)-1
    error_small(:,pa(i):pa(i+1)-1) = Xb(:,pa(i):pa(i+1)-1)-Db*Gamma(:,pa(i):pa(i+1)-1);
    err_small(pa(i):pa(i+1)-1) = sum(error_small(:,pa(i):pa(i+1)-1).^2);
end
deItem = 0;
usecount = sum(abs(Gamma)>1e-7, 2);
for j = 1:ds
    Gj_big = Df'*Df(:,j);
    Gj_big(j) = 0;
    Gj_small = Db'*Db(:,j);
    Gj_small(j) = 0;
    % replace atom
    if ( (max(Gj_big.^2)>th^2 ||max(Gj_small.^2)>th^2|| usecount(j)<maxTh) && ~rItem(j) )
        [~,i] = max(diff(uI));
        Df(:,j) = Xf(:,uI(i)) / norm(Xf(:,uI(i)));  %
        Db(:,j) = Xb(:,uI(i)) / norm(Xb(:,uI(i)));
        uI = uI([1:i-1,i+1:end]);
        deItem = deItem+1;
    end
end

end




