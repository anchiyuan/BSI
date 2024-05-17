function source_MINT = MINT(a, y, g_len, weight_len, dia_load)
% a          : predicted RIR
% y          : microphone signal
% g_len      : length of RIR for filling into G
% weight_len : filter length for one channel
% dia_load   : factor for Tikhonov regularization

g = a(:, 1:g_len);
%%%%%%%%%%%%%%%%%%% can change %%%%%%%%%%%%%%%%%%%
mic_choose = [4, 1, 6, 3, 5, 2];
num_channel = 6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
g_choose = g(mic_choose, :);
[g_choose_max_value, g_choose_max_index] = max(abs(g_choose), [], 2);
[max_index, max_value_index] = max(g_choose_max_index);

D = zeros(g_len+weight_len-1, 1);
D_delay = 0;
D(max_index + D_delay, :) = 1;
fprintf('peak should appear at %d\n', max_index + D_delay)

G = zeros(g_len+weight_len-1, num_channel*weight_len);
for i = 1 : weight_len
    for ii = 1 : num_channel
        G(i:(g_len+i-1), (ii - 1)*weight_len + i) = g(mic_choose(ii), :).';
    end
end
ran = rank(G);
size_G = min(size(G));
fprintf('shape of G = %d rank of G = %d\n', size_G, ran)

W = inv(G'*G + dia_load*eye(size(G, 2)))*G'*D;

SorLen = size(y, 2);
source_MINT = zeros(1, weight_len+SorLen-1);
for i = 1 : num_channel
    source_MINT = source_MINT + conv(W((i-1)*weight_len+1:i*weight_len, :).', y(mic_choose(i), :));
end
source_MINT = source_MINT(1:SorLen);
