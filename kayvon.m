N = 50;
w = randn(N)*1.1;
w = w/max(real(eig(w)))*.9;
dt = 0.01;
tau = .1;
t = 0:dt:.5;
fun = @(x) tanh(x);
inp = randn(1, N);
cn = 50;
alpha = .005;
alphai = .005;
clf
for iter = 1:100;
    r = zeros(1,N);
    for i = 1:length(t)-1
        r(i+1,:) = r(i,:) + dt/tau*(-r(i,:) + fun(r(i,:))*w + inp*(i==10));
    end

    if iter == 1;
        [a,b] = sort(sum(abs(r)));
        cn = b(1);
    end

    r = fun(r);
    rew = r(:,cn);
    dw = corr(r,rew);
    di = dw;
    dw = dw*dw';
    dw = dw - eye(N).*dw;
    dw = dw*sum(rew);
    w = w + dw*alpha/norm(w);
    inp = inp + di'*alphai;
    plot(r(:,cn));
    drawnow;
    E(iter) = sum(r(:,cn));
    R(:,:,iter) = r;
end