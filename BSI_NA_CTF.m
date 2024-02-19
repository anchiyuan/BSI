clc; clear;
close all;

tic

%% RIR parameter %%
SorNum = 1;                                              % source number
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MicNum = 3;                                             % number of microphone
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

% window parameters %
NFFT = 64;
hopsize = 16;

win = hamming(NFFT);
osfac = round(NFFT/hopsize);

frequency = NFFT/2 + 1;
L = length(hopsize:hopsize:points_rir+2*NFFT-2);

% delay as %
extra_delay_x = (ceil(NFFT/hopsize) - 1)*hopsize;    % put delay for equilization between time convolution and CTF 
x_delay = zeros(MicNum, SorLen);
x_delay(:, extra_delay_x+1:end) = as(:, 1:SorLen-extra_delay_x);

% STFT %
[X, ~, ~] = stft(x_delay.', fs, Window=win, OverlapLength=NFFT-hopsize, FFTLength=NFFT, FrequencyRange='onesided');
NumOfFrame = size(X, 2);
NumOfFrame_vector = 1:1:NumOfFrame;

%% NMCCTFLMS %%
% basic parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = 0.999;
step_size = 0.999;
diaload_delta_weight = 1e-5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize H_hat %
H_hat = zeros(MicNum, L, frequency);

% initialize cost function %
cost_fun = zeros(frequency, NumOfFrame);

% independent frequency parfor loop %
for n = 1:frequency
    % initialize R %
    R_ini_scalar = 0;
    for i = 1:MicNum
        R_ini_scalar = R_ini_scalar + flip(conj(X(n, L:2*L-1, i)))*flip(X(n, L:2*L-1, i)).'/L;    % X 要到第 L-osfac 個 frame 才能用
    end

    R = R_ini_scalar*eye(MicNum*L);

    % initialize error %
    error = zeros(NumOfFrame, 1);
    
    % initialize weight %
    weight = zeros(MicNum*L, 1);
    for i = 1:MicNum
        weight((i-1)*L+osfac, :) = 1;
    end

    weight = weight/norm(weight);
    
    % iteration process %
    for FrameNo = 2*L:NumOfFrame
        % update R %
        R_temp = zeros(MicNum*L, MicNum*L);
        dia_sum = zeros(L, L);
        for i = 1:MicNum
            for j = 1:MicNum
                R_temp((i-1)*L+1:i*L, (j-1)*L+1:j*L) = -(flip(X(n, FrameNo-L+1:FrameNo, j)).'*flip(conj(X(n, FrameNo-L+1:FrameNo, i))));
                if i == j
                    dia_sum = dia_sum + flip(X(n, FrameNo-L+1:FrameNo, j)).'*flip(conj(X(n, FrameNo-L+1:FrameNo, i)));
                end
    
            end
    
        end
    
        for i = 1:MicNum
            R_temp((i-1)*L+1:i*L, (i-1)*L+1:i*L) = R_temp((i-1)*L+1:i*L, (i-1)*L+1:i*L) + dia_sum;
        end
    
        R = lambda*R + R_temp;
  
        % update error %
        for i = 1:MicNum-1
            for j = i+1:MicNum
                error(FrameNo, :) = error(FrameNo, :) + abs(flip(conj(X(n, FrameNo-L+1:FrameNo, i)))*weight((j-1)*L+1:j*L, :) - flip(conj(X(n, FrameNo-L+1:FrameNo, j)))*weight((i-1)*L+1:i*L, :))^2;
            end
    
        end

        % update weight %
        delta_weight = inv(R-2*weight*(weight')*R-2*R*weight*(weight')+diaload_delta_weight*eye(MicNum*L))*(R_temp*weight-error(FrameNo, :)*weight);
        weight = (weight-step_size*delta_weight)/norm(weight-step_size*delta_weight);

    end
    
    % reshape and save %
    H_hat(:, :, n) = reshape(weight, [L MicNum]).';
    cost_fun(n, :) = error;
    
end

% 轉回時域 %
H_hat_forplot = zeros(frequency, L, MicNum);
for i = 1 : MicNum
    H_hat_forplot(:, :, i) = squeeze(H_hat(i, :, :)).';
end

h_hat = reconstruct_RIR_normalwindow(points_rir, NFFT, hopsize, L, win, fs, frequency, MicNum, H_hat_forplot);    % dimension = MicNum x (points_rir+(osfac-1)*hopsize)
ratio_h_hat = zeros(MicNum, 1);
for i = 1:MicNum
    ratio_h_hat(i, :) = max(abs(h(i, :)))/max(abs(h_hat(i, hopsize*(osfac-1)+1:end)));
end

h_hat = h_hat(:, hopsize*(osfac-1)+1:end).*ratio_h_hat;

% 畫圖看結果 %
figure(3)
plot(mean(cost_fun(:, 2*L:end)));
xlabel('update times')
title('cost function')

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
