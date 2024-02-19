clc; clear;
close all;

% 加入資料夾 %
addpath('wpe_v1.33')

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
g_len = 128;
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

%% load ground-truth RIR (h) %%
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

%% window parameters %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NFFT = 64;
hopsize = 16;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

win = hamming(NFFT);
osfac = round(NFFT/hopsize);

frequency = NFFT/2 + 1;
L = length(hopsize:hopsize:points_rir+2*NFFT-2);

%% 讀音檔 (source) %%
Second = 23;
SorLen =  Second*fs;

% load source %
% [source_transpose, fs] = audioread('245.wav', [1, SorLen]);
% source = source_transpose.';

source = wgn(1, SorLen, 0);

%% RIR mix source 先在時域上 convolution 再做 stft (y_delay y_nodelay Y_delay) %%
% convolution source and RIR %
as = zeros(MicNum, points_rir+SorLen-1);
for i = 1 : MicNum
    as(i, :) = conv(h(i, :), source);
end

extra_delay_y = (ceil(NFFT/hopsize) - 1)*hopsize;    % put delay for equilization between time convolution and CTF 
y_delay = zeros(MicNum, SorLen);
y_delay(:, extra_delay_y+1:end) = as(:, 1:SorLen-extra_delay_y);
y_nodelay = as(:, 1:SorLen);

% y 轉頻域 %
y_delay_transpose = y_delay.';
[Y_delay, ~, ~] = stft(y_delay_transpose, fs, Window=win, OverlapLength=NFFT-hopsize, FFTLength=NFFT, FrequencyRange='onesided');

%% load y_wpe %%
% do wpe %
y_wpe = wpe(y_nodelay.', 'wpe_parameter.m');
y_wpe = y_wpe.';

% 存 wpe mat %
y_wpe_filename_str = ['y\y_wpe-', string(reverberation_time), '.mat'];
y_wpe_filename = join(y_wpe_filename_str, '');
save(y_wpe_filename, 'y_wpe')

% % 存 wpe wav %
% ratio_y_wpe = 0.8 / max(abs(y_wpe(look_mic, :))) ;
% y_filemane_str = ['wav\y_wpe_all-', string(reverberation_time), '.wav'];
% y_filemane = join(y_filemane_str, '');
% audiowrite(y_filemane, y_wpe(look_mic, :)*ratio_y_wpe, fs)

% load y_wpe %
y_wpe_filename_str = ['y\y_wpe-', string(reverberation_time), '.mat'];
y_wpe_filename = join(y_wpe_filename_str, '');
load(y_wpe_filename);

%% DAS beamformer %%
% 算 mic 與 source 之距離 %
distance = zeros(MicNum, SorNum);
for i = 1 : MicNum
    distance(i, :) =  sqrt(sum((SorPos - MicPos(i, :)).^2));
end

% 算 a %
a = zeros(MicNum, SorNum, frequency);
freqs_vector = linspace(0, fs/2, frequency);
for n = 1:frequency
    omega = 2*pi*freqs_vector(n);
    a(:, :, n) = exp(-1j*omega/c*distance)./distance;
end

% 算 DAS weight %
w = a/MicNum;

% 算 Y_DAS %
y_wpe_transpose = y_wpe.';
[Y_wpe, ~, ~] = stft(y_wpe_transpose, fs, Window=win, OverlapLength=NFFT-hopsize, FFTLength=NFFT, FrequencyRange='onesided');

NumOfFrame = size(Y_wpe, 2);
Y_DAS = zeros(frequency, NumOfFrame);
for FrameNo= 1:NumOfFrame
    for n = 1: frequency
         Y_DAS(n, FrameNo) = w(:, :, n)'*squeeze(Y_wpe(n, FrameNo, :));
    end  

end

%%  Wiener filter %%
start_ini_frame = L;
ini_frame = NumOfFrame;

% compute Rss %
forfac_ini_Rss = 0.999;
Rss = zeros(L, L, frequency);
for FrameNo= start_ini_frame:ini_frame
    S_choose = Y_DAS(:, FrameNo+1-L:FrameNo).';
    for n = 1: frequency
         Rss(:,:,n) = forfac_ini_Rss*Rss(:,:,n) + (1-forfac_ini_Rss)*flip(S_choose(:,n)) * flip(S_choose(:,n))';
    end  

end

% compute Rsy %
forfac_ini_Rsy = 0.999;
Rsy = zeros(L, MicNum, frequency);
for FrameNo= start_ini_frame:ini_frame
    S_choose = Y_DAS(:, FrameNo+1-L:FrameNo).';
    Y_choose = squeeze(Y_delay(:, FrameNo, :)).';
    for n = 1: frequency
         Rsy(:,:,n) = forfac_ini_Rsy*Rsy(:,:,n) + (1-forfac_ini_Rsy)*flip(S_choose(:,n)) * Y_choose(:,n)';
    end 

end

% Wiener solution %
A = zeros(MicNum, L, frequency);
dia_load_ini = 10^(-7);
for n = 1:frequency
    A(:,:,n) = Rsy(:,:,n)'/(Rss(:,:,n) + dia_load_ini.*eye(L));
end

% save A %
algorithm = 'Wiener';
A_filename_str = ['A\A-', string(reverberation_time), 'x', algorithm, '.mat'];
A_filename = join(A_filename_str, '');
save(A_filename, 'A')

% A_tdomain %
A_forplot = zeros(frequency, L, MicNum);
for i = 1 : MicNum
    A_forplot(:, :, i) = squeeze(A(i, :, :)).';
end

A_tdomain = reconstruct_RIR_normalwindow(points_rir, NFFT, hopsize, L, win, fs, frequency, MicNum, A_forplot);    % dimension = MicNum x (points_rir+(osfac-1)*hopsize)
ratio_A_tdomain = zeros(MicNum, 1);
for i = 1:MicNum
    ratio_A_tdomain(i, :) = max(abs(h(i, :)))/max(abs(A_tdomain(i, hopsize*(osfac-1)+1:end)));
end

A_tdomain = A_tdomain(:, hopsize*(osfac-1)+1:end).*ratio_A_tdomain;

% 畫圖看結果 %
look_mic = 1;
figure(3)
plot(h(look_mic, :), 'r');
hold on
plot(A_tdomain(look_mic, :), 'b');
hold off
xlim([1 points_rir])
legend('ground-truth RIR', 'estimated RIR')
xlabel('time samples')
ylabel('amplitude')
shg

% ME %
ATF = fft(h, points_rir, 2);
ATF_estimated = fft(A_tdomain, points_rir, 2);
sum_norm = 0;
for i  = 1:MicNum
    norm_ATF = norm(ATF(i, :) - ATF_estimated(i, :));
    sum_norm = sum_norm + norm_ATF;
end

ME_Wiener = sum_norm/MicNum;

% NRMSPM %
h_NRMSPM = reshape(h.', [MicNum*points_rir 1]);
aa_NRMSPM = reshape(A_tdomain.', [MicNum*points_rir 1]);
NRMSPM_Wiener = 20*log(norm(h_NRMSPM-h_NRMSPM.'*aa_NRMSPM/(aa_NRMSPM.'*aa_NRMSPM)*aa_NRMSPM)/norm(h_NRMSPM));

% % MINT 存音檔 %
% A_tdomain_forMINT = A_tdomain;
% weight_len = floor(g_len/4);
% dia_load_MINT = 10^(-7);
% source_MINT_Wiener = MINT(A_tdomain_forMINT, y_nodelay, g_len, weight_len, dia_load_MINT);

%% RLS %%
start_ini_frame = L;
ini_frame = NumOfFrame;

% vector error RLS %
A = zeros(MicNum, L, frequency);
error = zeros(MicNum, (ini_frame-start_ini_frame+1), frequency);
lambda = 0.99;
for n = 1: frequency
    weight = zeros(L, MicNum);
    P = (1e-2)^-1*eye(L);
    for FrameNo = start_ini_frame:ini_frame
        Y_DAS_choose = flip(Y_DAS(n, FrameNo-L+1:FrameNo).');
        Y_delay_choose = squeeze(Y_delay(n, FrameNo, :));
        error(:, FrameNo-start_ini_frame+1, n) = Y_delay_choose' - Y_DAS_choose'*weight;
        k_RLS = lambda^-1*P*Y_DAS_choose/(1 + lambda^-1*Y_DAS_choose'*P*Y_DAS_choose);
        weight = weight + k_RLS*error(:, FrameNo-start_ini_frame+1, n).';
        P = lambda^-1*P - lambda^-1*k_RLS*Y_DAS_choose'*P;
    end

    A(:, :, n) = weight';
end

% save A %
algorithm = 'RLS';
A_filename_str = ['A\A-', string(reverberation_time), 'x', algorithm, '.mat'];
A_filename = join(A_filename_str, '');
save(A_filename, 'A')

% A_tdomain %
A_forplot = zeros(frequency, L, MicNum);
for i = 1 : MicNum
    A_forplot(:, :, i) = squeeze(A(i, :, :)).';
end

A_tdomain = reconstruct_RIR_normalwindow(points_rir, NFFT, hopsize, L, win, fs, frequency, MicNum, A_forplot);    % dimension = MicNum x (points_rir+(osfac-1)*hopsize)
ratio_A_tdomain = zeros(MicNum, 1);
for i = 1:MicNum
    ratio_A_tdomain(i, :) = max(abs(h(i, :)))/max(abs(A_tdomain(i, hopsize*(osfac-1)+1:end)));
end

A_tdomain = A_tdomain(:, hopsize*(osfac-1)+1:end).*ratio_A_tdomain;

% 畫圖看結果 %
look_mic = 1;
figure(4)
plot(h(look_mic, :), 'r');
hold on
plot(A_tdomain(look_mic, :), 'b');
hold off
xlim([1 points_rir])
legend('ground-truth RIR', 'estimated RIR')
xlabel('time samples')
ylabel('amplitude')
shg

% ME %
ATF = fft(h, points_rir, 2);
ATF_estimated = fft(A_tdomain, points_rir, 2);
sum_norm = 0;
for i  = 1:MicNum
    norm_ATF = norm(ATF(i, :) - ATF_estimated(i, :));
    sum_norm = sum_norm + norm_ATF;
end

ME_RLS = sum_norm/MicNum;

% NRMSPM %
h_NRMSPM = reshape(h.', [MicNum*points_rir 1]);
aa_NRMSPM = reshape(A_tdomain.', [MicNum*points_rir 1]);
NRMSPM_RLS = 20*log(norm(h_NRMSPM-h_NRMSPM.'*aa_NRMSPM/(aa_NRMSPM.'*aa_NRMSPM)*aa_NRMSPM)/norm(h_NRMSPM));

% % MINT 存音檔 %
% A_tdomain_forMINT = A_tdomain;
% weight_len = floor(g_len/4);
% dia_load_MINT = 10^(-7);
% source_MINT_RLS = MINT(A_tdomain_forMINT, y_nodelay, g_len, weight_len, dia_load_MINT);

%% Kalman filter %%
start_ini_frame = L;
ini_frame = NumOfFrame;

% Kalman filter %
A = zeros(MicNum, L, frequency);
parfor i = 1:MicNum
    for n =1:frequency
        weight = zeros(L, 1);
        P = 0.5*eye(L);    % error covariance matrix %
        K = zeros(L, 1);    % Kalman gain %
        R = 10^(-3);    % measurement noise covariance matrix %
        for FrameNo = start_ini_frame:ini_frame
            % no time update only have measurement update %
            K = P*flip(Y_DAS(n, FrameNo-L+1:FrameNo).')*inv(conj(flip(Y_DAS(n, FrameNo-L+1:FrameNo)))*P*flip(Y_DAS(n, FrameNo-L+1:FrameNo).') + R);
            weight = weight + K*(conj(Y_delay(n, FrameNo, i)) - conj(flip(Y_DAS(n, FrameNo-L+1:FrameNo)))*weight);
            P = P - K*conj(flip(Y_DAS(n, FrameNo-L+1:FrameNo)))*P;
        end
    
        A(i, :, n) = weight';
    end

end

% save A %
algorithm = 'Kalman';
A_filename_str = ['A\A-', string(reverberation_time), 'x', algorithm, '.mat'];
A_filename = join(A_filename_str, '');
save(A_filename, 'A')

% A_tdomain %
A_forplot = zeros(frequency, L, MicNum);
for i = 1 : MicNum
    A_forplot(:, :, i) = squeeze(A(i, :, :)).';
end

A_tdomain = reconstruct_RIR_normalwindow(points_rir, NFFT, hopsize, L, win, fs, frequency, MicNum, A_forplot);    % dimension = MicNum x (points_rir+(osfac-1)*hopsize)
ratio_A_tdomain = zeros(MicNum, 1);
for i = 1:MicNum
    ratio_A_tdomain(i, :) = max(abs(h(i, :)))/max(abs(A_tdomain(i, hopsize*(osfac-1)+1:end)));
end

A_tdomain = A_tdomain(:, hopsize*(osfac-1)+1:end).*ratio_A_tdomain;

% 畫圖看結果 %
look_mic = 1;
figure(5)
plot(h(look_mic, :), 'r');
hold on
plot(A_tdomain(look_mic, :), 'b');
hold off
xlim([1 points_rir])
legend('ground-truth RIR', 'estimated RIR')
xlabel('time samples')
ylabel('amplitude')
shg

% ME %
ATF = fft(h, points_rir, 2);
ATF_estimated = fft(A_tdomain, points_rir, 2);
sum_norm = 0;
for i  = 1:MicNum
    norm_ATF = norm(ATF(i, :) - ATF_estimated(i, :));
    sum_norm = sum_norm + norm_ATF;
end

ME_Kalman = sum_norm/MicNum;

% NRMSPM %
h_NRMSPM = reshape(h.', [MicNum*points_rir 1]);
aa_NRMSPM = reshape(A_tdomain.', [MicNum*points_rir 1]);
NRMSPM_Kalman = 20*log(norm(h_NRMSPM-h_NRMSPM.'*aa_NRMSPM/(aa_NRMSPM.'*aa_NRMSPM)*aa_NRMSPM)/norm(h_NRMSPM));

% % MINT 存音檔 %
% A_tdomain_forMINT = A_tdomain;
% weight_len = floor(g_len/4);
% dia_load_MINT = 10^(-7);
% source_MINT_Kalman = MINT(A_tdomain_forMINT, y_nodelay, g_len, weight_len, dia_load_MINT);

%% save .wav 檔 %%
point_start_save = 18*fs;
source_max  = max(abs(source(1, point_start_save:end)));
source_MINT_Wiener_max  = max(abs(source_MINT_Wiener(1, point_start_save:end)));
source_MINT_RLS_max  = max(abs(source_MINT_RLS(1, point_start_save:end)));
source_MINT_Kalman_max  = max(abs(source_MINT_Kalman(1, point_start_save:end)));

audiowrite('wav\source.wav', source(1, point_start_save:end), fs)

ratio_y_nodelay = 0.8 / max(abs(y_nodelay(look_mic, point_start_save:end))) ;
y_filemane_str = ['wav\y_nodelay-', string(reverberation_time), '.wav'];
y_filemane = join(y_filemane_str, '');
audiowrite(y_filemane, y_nodelay(look_mic, point_start_save:end)*ratio_y_nodelay, fs)

ratio_y_wpe = 0.8 / max(abs(y_wpe(look_mic, point_start_save:end))) ;
y_filemane_str = ['wav\y_wpe-', string(reverberation_time), '.wav'];
y_filemane = join(y_filemane_str, '');
audiowrite(y_filemane, y_wpe(look_mic, point_start_save:end)*ratio_y_wpe, fs)

source_MINT_filemane_str = ['wav\source_predict_Wiener-', string(reverberation_time), '.wav'];
source_MINT_filemane = join(source_MINT_filemane_str, '');
audiowrite(source_MINT_filemane, source_MINT_Wiener(1, point_start_save:end).*(source_max/source_MINT_Wiener_max), fs)

source_MINT_filemane_str = ['wav\source_predict_Wiener-', string(reverberation_time), '.wav'];
source_MINT_filemane = join(source_MINT_filemane_str, '');
audiowrite(source_MINT_filemane, source_MINT_Wiener(1, point_start_save:end).*(source_max/source_MINT_RLS_max), fs)

source_MINT_filemane_str = ['wav\source_predict_Wiener-', string(reverberation_time), '.wav'];
source_MINT_filemane = join(source_MINT_filemane_str, '');
audiowrite(source_MINT_filemane, source_MINT_Wiener(1, point_start_save:end).*(source_max/source_MINT_Kalman_max), fs)

fprintf('done\n')
