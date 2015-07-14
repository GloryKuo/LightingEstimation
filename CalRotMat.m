GG = @(A,B) [ dot(A,B) -norm(cross(A,B)) 0;
              norm(cross(A,B)) dot(A,B)  0;
              0              0           1];

FFi = @(A,B) [ A (B-dot(A,B)*A)/norm(B-dot(A,B)*A) cross(B,A) ];

UU = @(Fi,G) Fi*G*inv(Fi);

a=[1 0 0]'; b=[0 1 0]';
U = UU(FFi(a,b), GG(a,b));
norm(U) % is it length-preserving?
%ans = 1
norm(b-U*a) % does it rotate a onto b?
%ans = 0
U
% U =
% 
%    0  -1   0
%    1   0   0
%    0   0   1