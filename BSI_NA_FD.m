clc; clear;
close all;

tic

%% RIR parameter %%
SorNum = 1;                                              % source number
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MicNum = 6;                                             % number of microphone
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c = 343;                                                 % Sound velocity (m/s)
fs = 16000;                                              % Sample frequency (samples/s)
Ts = 1/fs;                                               % Sample period (s)

% ULA %
MicStart = [0.1, 0.1, 0.2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
spacing = 0.02;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MicPos = zeros(MicNum, 3);
for i = 1:MicNum
    MicPos(i, :) = [MicStart(1, 1)+(i-1)*spacing, MicStart(1, 2), MicStart(1, 3)];
end

SorPos = [0.2, 0.3, 0.2];                                % source position (m)
room_dim = [0.3, 0.4, 0.3];                              % Room dimensions [x y z] (m)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reverberation_time = 0.009;                               % Reverberation time (s)
points_rir = 256;                                        % Number of rir points (需比 reverberation time 還長)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mtype = 'omnidirectional';                               % Type of microphone
order = -1;                                              % -1 equals maximum reflection order!
dim = 3;                                                 % Room dimension
orientation = 0;                                         % Microphone orientation (rad)
hp_filter = 1;                                           % Disable high-pass filter

% 畫空間圖 %
figure(1);
plot3( [0 room_dim(1, 1) room_dim(1, 1) 0 0 0 room_dim(1, 1) room_dim(1, 1) 0 0 room_dim(1, 1) room_dim(1, 1) 0 0 room_dim(1, 1) room_dim(1, 1)], ...
       [0 0 room_dim(1, 2) room_dim(1, 2) 0 0 0 room_dim(1, 2) room_dim(1, 2) room_dim(1, 2) room_dim(1, 2) room_dim(1, 2) room_dim(1, 2) 0 0 0], ...
       [0 0 0 0 0 room_dim(1, 3) room_dim(1, 3) room_dim(1, 3) room_dim(1, 3) 0 0 room_dim(1, 3) room_dim(1, 3) room_dim(1, 3) room_dim(1, 3) 0] , 'k')
hold on
plot3(MicPos(:, 1), MicPos(:, 2), MicPos(:, 3), 'r.', 'MarkerSize', 10)
hold on
plot3(SorPos(:, 1), SorPos(:, 2), SorPos(:, 3), '*', 'MarkerSize', 20)
hold off
xlabel('x\_axis')
ylabel('y\_axis')
zlabel('z\_axis')
title('空間圖')
shg

%% generate ground-truth RIR (h) %%
% 產生 RIR 和存.mat 檔 %
h = rir_generator(c, fs, MicPos, SorPos, room_dim, reverberation_time, points_rir, mtype, order, dim, orientation, hp_filter);
rir_filename_str = ['h\h_', string(reverberation_time), 'x', string(MicNum), 'x', string(points_rir), '.mat'];
rir_filemane = join(rir_filename_str, '');
save(rir_filemane, 'h')

% load RIR 的 .mat 檔 %
rir_filename_str = ['h\h_', string(reverberation_time), 'x', string(MicNum), 'x', string(points_rir), '.mat'];
rir_filemane = join(rir_filename_str, '');
load(rir_filemane)

look_mic = 1;
h_yaxis_upperlimit = max(h(look_mic, :)) + 0.01;
h_yaxis_underlimit = min(h(look_mic, :)) - 0.01;
% 畫 ground-truth RIR time plot %
figure(2)
plot(h(look_mic, :), 'r');
xlim([1 points_rir])
ylim([h_yaxis_underlimit h_yaxis_upperlimit])
title('h')
xlabel('points')
ylabel('amplitude')
shg

%% 讀音檔 or 產生 white noise source (source) %%
Second = 23;
SorLen =  Second*fs;

% load source %
% [source_transpose, fs] = audioread('245.wav', [1, SorLen]);    % speech source
% source = source_transpose.';

source = wgn(1, SorLen, 0);

%% RIR mix source 先在時域上 convolution 再做 stft (x and X) %%
% convolution source and RIR %
as = zeros(MicNum, points_rir+SorLen-1);
for i = 1 : MicNum
    as(i, :) = conv(h(i, :), source);
end

x = as(:, 1:SorLen);

%% NMCFLMS %%
% basic parameters %
L = points_rir;
total_block = floor((SorLen-2*L)/L)+1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = (1-1/(3*L))^L;    % (1-1/(3*L))^L
step_size = 0.1;
dia_load = 0.1;    % mean(x(:, 1:2*L).^2, "all")/5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% basic matrix %
F_LxL = zeros(L, L);
for p = 1:L
    for q = 1:L
        F_LxL(p, q) = exp(-1j*2*pi*(p-1)*(q-1)/L);
    end
end

F_2Lx2L = zeros(2*L, 2*L);
for p = 1:2*L
    for q = 1:2*L
        F_2Lx2L(p, q) = exp(-1j*2*pi*(p-1)*(q-1)/(2*L));
    end
end

W_01_Lx2L = F_LxL*[zeros(L, L), eye(L)]*inv(F_2Lx2L);
W_10_2LxL = F_2Lx2L*[eye(L), zeros(L, L)].'*inv(F_LxL);
W_10_Lx2L = F_LxL*[eye(L), zeros(L, L)]*inv(F_2Lx2L);
W_01_2LxL = F_2Lx2L*[zeros(L, L), eye(L)].'*inv(F_LxL);

% initialize h_hat, h_hat_ul and h_hat_ul_10 %
h_hat = zeros(MicNum*points_rir, 1);
for i = 1:MicNum
    h_hat((i-1)*points_rir+1) = 1;
end

h_hat = h_hat/norm(h_hat);
h_hat = reshape(h_hat, [L MicNum]);

h_hat_ul = F_LxL*h_hat;
h_hat_ul_10 = W_10_2LxL*h_hat_ul;

% initialize e_ul and e_ul_01 %
e_ul = zeros(MicNum, MicNum, L);
e_ul_01 = zeros(MicNum, MicNum, 2*L);
e_mean = zeros(total_block, 1);

% initialize D %
D = zeros(MicNum, 2*L, 2*L);

% initialize P %
P = zeros(MicNum, 2*L, 2*L);

% iteration process %
for b = 2:total_block
    % update D %
    for i = 1:MicNum
        D(i, :, :) = diag(fft(x(i, (b-1)*L+1:(b-1)*L+2*L).', 2*L));
    end

    % update e_ul_01 %
    for i = 1:MicNum
        for j = 1:MicNum
            e_ul(i, j, :) = W_01_Lx2L*(squeeze(D(i, :, :))*W_10_2LxL*h_hat_ul(:, j)-squeeze(D(j, :, :))*W_10_2LxL*h_hat_ul(:, i));
            e_ul_01(i, j, :) = W_01_2LxL*squeeze(e_ul(i, j, :));
        end
    end

    e_mean(b, 1) = mean(abs(e_ul), 'all');

    % update P %
    for i = 1:MicNum
        P_accumulate = zeros(2*L, 2*L);
        for j = 1:MicNum
            if j ~= i
                P_accumulate = P_accumulate + squeeze(conj(D(j, :, :)))*squeeze(D(j, :, :));
            end
        end

        P(i, :, :) = lambda*squeeze(P(i, :, :)) + (1-lambda)*P_accumulate;
    end

    % update h_hat_ul_10 and h_hat_ul %
    for i = 1:MicNum
        De_accumulate = zeros(2*L, 1);
        for j = 1:MicNum
            De_accumulate = De_accumulate + squeeze(conj(D(j, :, :)))*squeeze(e_ul_01(j, i, :));
        end

        h_hat_ul_10(:, i) = h_hat_ul_10(:, i) - step_size*inv(squeeze(P(i, :, :)) + dia_load*eye(2*L))*De_accumulate;
        h_hat_ul(:, i) = W_10_Lx2L*h_hat_ul_10(:, i);
    end
    
end

h_hat = real((inv(F_LxL)*h_hat_ul).');    % 虛部數值其實是零 (數值問題)
% rescale h_hat %
ratio_h_hat = zeros(MicNum, 1);
for i = 1:MicNum
    ratio_h_hat(i, :) = max(abs(h(i, :)))/max(abs(h_hat(i, :)));
end

h_hat = h_hat.*ratio_h_hat;

% 畫圖看結果 %
figure(3)
plot(2:1:total_block, e_mean(2:end, :));
xlabel('update blocks')
title('error')

look_mic = 1;
figure(4)
plot(h(look_mic, :), 'r');
hold on
plot(h_hat(look_mic, :), 'b');
hold off
xlim([1 points_rir])
legend('ground-truth RIR', 'estimated RIR')
xlabel('time samples')
ylabel('amplitude')
shg

% ME %
ATF = fft(h, points_rir, 2);
ATF_estimated = fft(h_hat, points_rir, 2);
sum_norm = 0;
for i  = 1:MicNum
    norm_ATF = norm(ATF(i, :) - ATF_estimated(i, :));
    sum_norm = sum_norm + norm_ATF;
end

ME = sum_norm/MicNum;

% NRMSPM %
h_NRMSPM = reshape(h.', [MicNum*points_rir 1]);
h_hat_NRMSPM = reshape(h_hat.', [MicNum*points_rir 1]);
NRMSPM = 20*log(norm(h_NRMSPM-h_NRMSPM.'*h_hat_NRMSPM/(h_hat_NRMSPM.'*h_hat_NRMSPM)*h_hat_NRMSPM)/norm(h_NRMSPM));

toc
