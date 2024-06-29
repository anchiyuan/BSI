clc; clear;
close all;

% 加入資料夾 %
addpath('wpe_v1.33')

tic

%% RIR parameter %%
SorNum = 1;                                              % source number
MicNum_TDOA = 8;                                         % TDOA麥克風數量
MicNum = 18;                                             % number of microphone
c = 343;                                                 % Sound velocity (m/s)
fs = 16000;                                              % Sample frequency (samples/s)

% distributed 8 mic %
mic_x = [ 100 ; 200 ; 200 ; 100 ; 100 ; 200 ; 200 ; 100 ]./100;
mic_y = [ 100 ; 100 ; 200 ; 200 ; 100 ; 100 ; 200 ; 200 ]./100;
mic_z = [ 100 ; 100 ; 100 ; 100 ; 200 ; 200 ; 200 ; 200 ]./100;
MicPos = [mic_x, mic_y, mic_z,];

% ULA 30 mics %
MicStart = [110, 100, 100]/100;
spacing = 0.02;
for i = MicNum_TDOA+1:MicNum
    MicPos(i, :) = [MicStart(1, 1)+(i-(MicNum_TDOA+1))*spacing, MicStart(1, 2), MicStart(1, 3)];
end

SorPos = [170, 180, 130]/100;                            % source position (m)
room_dim = [ 300, 300, 250]/100;                          % Room dimensions [x y z] (m)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reverberation_time = 0.1;                                % Reverberation time (s)
points_rir = 1024;                                       % Number of rir points (需比 reverberation time 還長)
look_mic = 18;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mtype = 'omnidirectional';                               % Type of microphone
order = -1;                                              % -1 equals maximum reflection order!
dim = 3;                                                 % Room dimension
orientation = 0;                                         % Microphone orientation (rad)
hp_filter = 1;                                           % Disable high-pass filter

referencce_point = MicPos(1, :);
sorpos_groundtruth = SorPos - referencce_point;

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
% % 產生 RIR 和存.mat 檔 %
% h = rir_generator(c, fs, MicPos, SorPos, room_dim, reverberation_time, points_rir, mtype, order, dim, orientation, hp_filter);
% rir_filename_str = ['h\h_', string(reverberation_time), 'x', string(MicNum), 'x', string(points_rir), '.mat'];
% rir_filemane = join(rir_filename_str, '');
% save(rir_filemane, 'h')

rir_filename_str = ['h\h_', string(reverberation_time), 'x', string(MicNum), 'x', string(points_rir), '.mat'];
rir_filemane = join(rir_filename_str, '');
load(rir_filemane)

%% window parameter %%
NFFT = 1024;
hopsize = 256;

win = hamming(NFFT);
osfac = round(NFFT/hopsize);

frequency = NFFT/2 + 1;
L = length(hopsize:hopsize:points_rir+2*NFFT-2);    % (len(win) + len(win) - 1) + points_rir - 1
freqs_vector = linspace(0, fs/2, frequency);

%% 讀音檔 (source) %%
Second = 23;
SorLen =  Second*fs;

% load speech source %
[source_transpose, fs] = audioread('245.wav', [1, SorLen]);
source = source_transpose.';

%% 產生麥克風訊號，先在時域上 convolution 再做 stft (y_nodelay y_delay Y_delay ) %%
% convolution source and RIR %
as = zeros(MicNum, points_rir+SorLen-1);
for i = 1 : MicNum
    as(i, :) = conv(h(i, :), source);
end

extra_delay_y = (ceil(NFFT/hopsize) - 1)*hopsize;    % put delay for equilization between time convolution and CTF 
y_delay = zeros(MicNum, SorLen);
y_delay(:, extra_delay_y+1:end) = as(:, 1:SorLen-extra_delay_y);
y_nodelay = as(:, 1:SorLen);

% y_delay 轉頻域 %
y_delay_transpose = y_delay.';
[Y_delay, ~, ~] = stft(y_delay_transpose, fs, Window=win, OverlapLength=NFFT-hopsize, FFTLength=NFFT, FrequencyRange='onesided');
NumOfFrame = size(Y_delay, 2);

%% TDOA localization %%
% GCC-PHAT for delay estimation %
delay = zeros(MicNum_TDOA-1, 1);
difference = zeros(MicNum_TDOA-1, 1);
for i = 1:MicNum_TDOA-1
    delay(i, :) = gccphat(y_nodelay(i+1,:).', y_nodelay(1,:).', fs);
    difference(i, :) = delay(i, :)*c;
end

% mics position with repect to reference point %
micpos = [mic_x, mic_y, mic_z,] - referencce_point;

% generate parameters matrix %
A = 2*[micpos(2:end, :), difference];
b = micpos(2:end, 1).^2 + micpos(2:end, 2).^2 + micpos(2:end, 3).^2 - difference.^2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma = 0.01;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psi = 4*sigma^2*(ones(MicNum_TDOA-1, 1)*ones(MicNum_TDOA-1, 1).' + eye(MicNum_TDOA-1))*difference*difference.';
dia_load_psi = 1;
invpsi = inv(psi+dia_load_psi*eye(MicNum_TDOA-1)); 
P = [1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 -1];

% find possible root %
syms Z
theta = pinv(A.'*invpsi*A+Z*P )*(A.'*invpsi*b);
I = theta.'*P*theta;
[num, den] = numden(I);
poly = sym2poly(num);
roots = roots(poly);
roots_real = roots(imag(roots)==0, :);    % 取只有實數的根
if isempty(roots_real)
    roots_real = [roots_real; 0];
end

% find actual root from posible root %
if size(roots_real, 1) > 1
    theta = zeros(4, size(roots_real, 1));
    costfun = zeros(size(roots_real, 1), 1);
    for i = 1:size(roots_real, 1)
        theta(:, i) = pinv(A.'*invpsi*A+roots_real(i, :)*P )*(A.'*invpsi*b);
        costfun(i, :) = (A*theta(:, i)-b).'*invpsi*(A*theta(:, i)-b);
    end

    [~, min_index] = min(costfun);
    sorpos_estimation = theta(1:3, min_index).';
else
    theta = pinv(A.'*invpsi*A+roots_real*P )*(A.'*invpsi*b);
    sorpos_estimation = theta(1:3, :).';
end

%% load y_wpe (y_wpe) %%
% % do wpe %
% y_wpe = wpe(y_nodelay.', 'wpe_parameter.m');
% y_wpe = y_wpe.';
% 
% % 存 wpe mat %
% y_wpe_filename_str = ['y\y_wpe_', string(reverberation_time), '.mat'];
% y_wpe_filename = join(y_wpe_filename_str, '');
% save(y_wpe_filename, 'y_wpe')

% load y_wpe %
y_wpe_filename_str = ['y\y_wpe_', string(reverberation_time), '.mat'];
y_wpe_filename = join(y_wpe_filename_str, '');
load(y_wpe_filename);


%% DAS beamformer (Y_DAS) %%
% 算 mic 與 source 之距離 %
distance = zeros(MicNum, SorNum);
for i = 1 : MicNum
    distance(i, :) =  sqrt(sum((SorPos - MicPos(i, :)).^2));    % 其實應該用sorpos_estimation+referencce_point 不過差不到一公分
end

% 算 a %
a = zeros(MicNum, SorNum, frequency);
for n = 1:frequency
    omega = 2*pi*freqs_vector(n);
    a(:, :, n) = exp(-1j*omega/c*distance)./distance;
end

% 算 DAS weight %
w = a/MicNum;

% 算 Y_DAS %
y_wpe_transpose = y_wpe.';
[Y_wpe, ~, ~] = stft(y_wpe_transpose, fs, Window=win, OverlapLength=NFFT-hopsize, FFTLength=NFFT, FrequencyRange='onesided');

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

% A 轉回時域 %
A_forplot = zeros(frequency, L, MicNum);
for i = 1 : MicNum
    A_forplot(:, :, i) = squeeze(A(i, :, :)).';
end

A_tdomain = reconstruct_RIR_normalwindow(points_rir, NFFT, hopsize, L, win, fs, frequency, MicNum, A_forplot);    % dimension = MicNum x (points_rir+(osfac-1)*hopsize)
A_tdomain = A_tdomain(:, hopsize*(osfac-1)+1:end);
ratio_A_tdomain = zeros(MicNum, 1);
for i = 1:MicNum
    ratio_A_tdomain(i, :) = max(abs(h(i, :)))/max(abs(A_tdomain(i, :)));
end

A_tdomain = A_tdomain.*ratio_A_tdomain;

%  算 NRMSPM %
h_NRMSPM = reshape(h.', [MicNum*points_rir 1]);
aa_NRMSPM = reshape(A_tdomain.', [MicNum*points_rir 1]);
NRMSPM_Wiener = 20*log10(norm(h_NRMSPM-h_NRMSPM.'*aa_NRMSPM/(aa_NRMSPM.'*aa_NRMSPM)*aa_NRMSPM)/norm(h_NRMSPM));

% plot and save estimated RIR and ATF fig %
algorithm = 'Wiener';
h_yaxis_upperlimit = max(h(look_mic, :)) + 0.01;
h_yaxis_underlimit = min(h(look_mic, :)) - 0.01;
ATF = fft(h, points_rir, 2);
ATF_estimated = fft(A_tdomain, points_rir, 2);

figure(1)
plot(h(look_mic, :), 'r');
hold on
plot(A_tdomain(look_mic, :), 'b');
hold off
ylim([h_yaxis_underlimit h_yaxis_upperlimit])
xlim([1 points_rir])
legend('ground-truth RIR', 'estimated RIR')
xlabel('time samples')
ylabel('RIR')

fig_filename_str = ['fig\CTF_combined_', string(reverberation_time), 'x', algorithm, '_RIR.fig'];
fig_filename = join(fig_filename_str, '');
savefig(fig_filename)

figure(2)
subplot(2, 1, 1);
semilogx(linspace(0, fs/2, points_rir/2+1), 20*log10(abs(ATF(look_mic, 1:points_rir/2+1))), 'r');
hold on
semilogx(linspace(0, fs/2, points_rir/2+1), 20*log10(abs(ATF_estimated(look_mic, 1:points_rir/2+1))), 'b');
hold off
xlim([200 8000])
legend('ground-truth ATF', 'estimated ATF')
xlabel('frequency (Hz)')
ylabel('dB')

subplot(2, 1, 2);
semilogx(linspace(0, fs/2, points_rir/2+1), unwrap(angle(ATF(look_mic, 1:points_rir/2+1))), 'r');
hold on
semilogx(linspace(0, fs/2, points_rir/2+1), unwrap(angle(ATF_estimated(look_mic, 1:points_rir/2+1))), 'b');
hold off
xlim([200 8000])
legend('ground-truth ATF', 'estimated ATF')
xlabel('frequency (Hz)')
ylabel('phase (radius)')

fig_filename_str = ['fig\CTF_combined_', string(reverberation_time), 'x', algorithm, '_ATF.fig'];
fig_filename = join(fig_filename_str, '');
savefig(fig_filename)

% save A_tdomain %
A_filename_str = ['A_tdomain\CTF_combined_', string(reverberation_time), 'x', algorithm, '_A_tdomain.mat'];
A_filename = join(A_filename_str, '');
save(A_filename, 'A_tdomain')

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

% A 轉回時域 %
A_forplot = zeros(frequency, L, MicNum);
for i = 1 : MicNum
    A_forplot(:, :, i) = squeeze(A(i, :, :)).';
end

A_tdomain = reconstruct_RIR_normalwindow(points_rir, NFFT, hopsize, L, win, fs, frequency, MicNum, A_forplot);    % dimension = MicNum x (points_rir+(osfac-1)*hopsize)
A_tdomain = A_tdomain(:, hopsize*(osfac-1)+1:end);
ratio_A_tdomain = zeros(MicNum, 1);
for i = 1:MicNum
    ratio_A_tdomain(i, :) = max(abs(h(i, :)))/max(abs(A_tdomain(i, :)));
end

A_tdomain = A_tdomain.*ratio_A_tdomain;

%  算 NRMSPM %
h_NRMSPM = reshape(h.', [MicNum*points_rir 1]);
aa_NRMSPM = reshape(A_tdomain.', [MicNum*points_rir 1]);
NRMSPM_RLS = 20*log10(norm(h_NRMSPM-h_NRMSPM.'*aa_NRMSPM/(aa_NRMSPM.'*aa_NRMSPM)*aa_NRMSPM)/norm(h_NRMSPM));

% plot and save estimated RIR and ATF fig %
algorithm = 'RLS';
h_yaxis_upperlimit = max(h(look_mic, :)) + 0.01;
h_yaxis_underlimit = min(h(look_mic, :)) - 0.01;
ATF = fft(h, points_rir, 2);
ATF_estimated = fft(A_tdomain, points_rir, 2);

figure(3)
plot(h(look_mic, :), 'r');
hold on
plot(A_tdomain(look_mic, :), 'b');
hold off
ylim([h_yaxis_underlimit h_yaxis_upperlimit])
xlim([1 points_rir])
legend('ground-truth RIR', 'estimated RIR')
xlabel('time samples')
ylabel('RIR')

fig_filename_str = ['fig\CTF_combined_', string(reverberation_time), 'x', algorithm, '_RIR.fig'];
fig_filename = join(fig_filename_str, '');
savefig(fig_filename)

figure(4)
subplot(2, 1, 1);
semilogx(linspace(0, fs/2, points_rir/2+1), 20*log10(abs(ATF(look_mic, 1:points_rir/2+1))), 'r');
hold on
semilogx(linspace(0, fs/2, points_rir/2+1), 20*log10(abs(ATF_estimated(look_mic, 1:points_rir/2+1))), 'b');
hold off
xlim([200 8000])
legend('ground-truth ATF', 'estimated ATF')
xlabel('frequency (Hz)')
ylabel('dB')

subplot(2, 1, 2);
semilogx(linspace(0, fs/2, points_rir/2+1), unwrap(angle(ATF(look_mic, 1:points_rir/2+1))), 'r');
hold on
semilogx(linspace(0, fs/2, points_rir/2+1), unwrap(angle(ATF_estimated(look_mic, 1:points_rir/2+1))), 'b');
hold off
xlim([200 8000])
legend('ground-truth ATF', 'estimated ATF')
xlabel('frequency (Hz)')
ylabel('phase (radius)')

fig_filename_str = ['fig\CTF_combined_', string(reverberation_time), 'x', algorithm, '_ATF.fig'];
fig_filename = join(fig_filename_str, '');
savefig(fig_filename)

% save A_tdomain %
A_filename_str = ['A_tdomain\CTF_combined_', string(reverberation_time), 'x', algorithm, '_A_tdomain.mat'];
A_filename = join(A_filename_str, '');
save(A_filename, 'A_tdomain')

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

% A 轉回時域 %
A_forplot = zeros(frequency, L, MicNum);
for i = 1 : MicNum
    A_forplot(:, :, i) = squeeze(A(i, :, :)).';
end

A_tdomain = reconstruct_RIR_normalwindow(points_rir, NFFT, hopsize, L, win, fs, frequency, MicNum, A_forplot);    % dimension = MicNum x (points_rir+(osfac-1)*hopsize)
A_tdomain = A_tdomain(:, hopsize*(osfac-1)+1:end);
ratio_A_tdomain = zeros(MicNum, 1);
for i = 1:MicNum
    ratio_A_tdomain(i, :) = max(abs(h(i, :)))/max(abs(A_tdomain(i, :)));
end

A_tdomain = A_tdomain.*ratio_A_tdomain;

%  算 NRMSPM %
h_NRMSPM = reshape(h.', [MicNum*points_rir 1]);
aa_NRMSPM = reshape(A_tdomain.', [MicNum*points_rir 1]);
NRMSPM_Kalman = 20*log10(norm(h_NRMSPM-h_NRMSPM.'*aa_NRMSPM/(aa_NRMSPM.'*aa_NRMSPM)*aa_NRMSPM)/norm(h_NRMSPM));

% plot and save estimated RIR and ATF fig %
algorithm = 'Kalman';
h_yaxis_upperlimit = max(h(look_mic, :)) + 0.01;
h_yaxis_underlimit = min(h(look_mic, :)) - 0.01;
ATF = fft(h, points_rir, 2);
ATF_estimated = fft(A_tdomain, points_rir, 2);

figure(5)
plot(h(look_mic, :), 'r');
hold on
plot(A_tdomain(look_mic, :), 'b');
hold off
ylim([h_yaxis_underlimit h_yaxis_upperlimit])
xlim([1 points_rir])
legend('ground-truth RIR', 'estimated RIR')
xlabel('time samples')
ylabel('RIR')

fig_filename_str = ['fig\CTF_combined_', string(reverberation_time), 'x', algorithm, '_RIR.fig'];
fig_filename = join(fig_filename_str, '');
savefig(fig_filename)

figure(6)
subplot(2, 1, 1);
semilogx(linspace(0, fs/2, points_rir/2+1), 20*log10(abs(ATF(look_mic, 1:points_rir/2+1))), 'r');
hold on
semilogx(linspace(0, fs/2, points_rir/2+1), 20*log10(abs(ATF_estimated(look_mic, 1:points_rir/2+1))), 'b');
hold off
xlim([200 8000])
legend('ground-truth ATF', 'estimated ATF')
xlabel('frequency (Hz)')
ylabel('dB')

subplot(2, 1, 2);
semilogx(linspace(0, fs/2, points_rir/2+1), unwrap(angle(ATF(look_mic, 1:points_rir/2+1))), 'r');
hold on
semilogx(linspace(0, fs/2, points_rir/2+1), unwrap(angle(ATF_estimated(look_mic, 1:points_rir/2+1))), 'b');
hold off
xlim([200 8000])
legend('ground-truth ATF', 'estimated ATF')
xlabel('frequency (Hz)')
ylabel('phase (radius)')

fig_filename_str = ['fig\CTF_combined_', string(reverberation_time), 'x', algorithm, '_ATF.fig'];
fig_filename = join(fig_filename_str, '');
savefig(fig_filename)

% save A_tdomain %
A_filename_str = ['A_tdomain\CTF_combined_', string(reverberation_time), 'x', algorithm, '_A_tdomain.mat'];
A_filename = join(A_filename_str, '');
save(A_filename, 'A_tdomain')

fprintf('done\n')

toc
