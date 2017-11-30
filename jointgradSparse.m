function [Gamma] = jointgradSparse(ep,Xf,Df,Xb,Db,iter)
if mod(iter,2)>0
    G = Df'*Df;
    for ii=1:size(Xf,2)
        B = -Df'*Xf(:,ii);
        sp = gradSparse(ep,G, B);
        Gamma(:,ii) = sp;
    end
else
    G = Db'*Db;
    for ii=1:size(Xb,2)
        B = -Db'*Xb(:,ii);
        sp = gradSparse(ep,G,B);
        Gamma(:,ii)=sp;
    end
end