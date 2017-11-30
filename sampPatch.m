function [FP, BP] = sampPatch(fIm, bIm, pz, pn)

[nrow, ncol] = size(fIm);

x = randperm(nrow-2*pz-1) + pz;
y = randperm(ncol-2*pz-1) + pz;

[X,Y] = meshgrid(x,y);

xrow = X(:);
ycol = Y(:);

if pn < length(xrow)
    xrow = xrow(1:pn);
    ycol = ycol(1:pn);
end

pn = length(xrow);

fIm = double(fIm);
bIm = double(bIm);

for ii = 1:pn
    row = xrow(ii);
    col = ycol(ii);
    
    Fpatch = fIm(row:row+pz-1,col:col+pz-1);
    Bpatch = bIm(row:row+pz-1,col:col+pz-1);
     
    FP(:,ii) = Fpatch(:)- mean(Fpatch(:));
    BP(:,ii) = Bpatch(:)- mean(Bpatch(:));
end