function [MH, MV] = find_dimensions(N)
% Find the dimensions of a rectangular array given the total number of antennas N
% The function returns MH (number of horizontal antennas) and MV (number of vertical antennas)

MH = floor(sqrt(N));
while mod(N, MH) ~= 0
  MH = MH - 1;
end
MV = N / MH;
end