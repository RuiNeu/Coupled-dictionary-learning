function [x] = gradSparse(ep,G,B)

G = double(G);
B = double(B);

th = 1e-8;
x = zeros(size(G, 1), 1);           

grad = G*sparse(x)+B;
[maxVal minVal]=max(abs(grad).*(x==0));

while 1   
  if grad(minVal)>ep+th,
    x(minVal)=(ep-grad(minVal))/G(minVal,minVal);
  elseif grad(minVal)<-ep-th,
    x(minVal)=(-ep-grad(minVal))/G(minVal,minVal);            
  else
    if all(x==0)
      break;
    end
  end    
  
  while 1
    a = x~=0;   
    Aa = G(a,a);
    ba = B(a);
    xa = x(a);

    temp = -ep*sign(xa)-ba; %softthreshold
    x_new= Aa\temp;
    idx = find(x_new);
    o_new=(temp(idx)/2 + ba(idx))'*x_new(idx) + ep*sum(abs(x_new(idx)));
    
    s = find(xa.*x_new<=0);
    if isempty(s)
      x(a)=x_new;
      loss=o_new;
      break;
    end
    x_min=x_new;
    o_min=o_new;
    d=x_new-xa;
    t=d./xa;
    for zd=s'
      x_s=xa-d/t(zd);
      x_s(zd)=0;  %make sure it's zero
      idx = find(x_s);
      o_s = (Aa(idx, idx)*x_s(idx)/2 + ba(idx))'*x_s(idx)+ep*sum(abs(x_s(idx)));
      if o_s<o_min,
        x_min=x_s;
        o_min=o_s;
      end
    end
    
    x(a)=x_min;
    loss=o_min;
  end 
    
  grad = G*sparse(x)+B;
  
  [maxVal minVal]=max(abs(grad).*(x==0));
  if maxVal <= ep+th,
    break;
  end
end