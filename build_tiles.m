function [gridx, gridv] = build_tiles(lby, uby,lbv,ubv, M, N)

off = [1; 3];
off = off./max(off);

dx = (uby - lby)/M;
TX = lby - dx:dx:uby;

dv = (ubv - lbv)/M;
TV = lbv - dv:dv:ubv;

gridx = zeros(N, length(TX));
gridv = zeros(N, length(TV));

gridx(1, :) = TX;
gridv(1, :) = TV;

for ii = 2 : N
    gridx(ii, :) = TX + off(1)*dx/N*(ii-1);
    gridv(ii, :) = TV + off(2)*dv/N*(ii-1);
end
