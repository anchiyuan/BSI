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
reverberation_time = 0.01;                               % Reverberation time (s)
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
% % 產生 RIR 和存.mat 檔 %
h = rir_generator(c, fs, MicPos, SorPos, room_dim, reverberation_time, points_rir, mtype, order, dim, orientation, hp_filter);
% rir_filename_str = ['h\h_', string(reverberation_time), 'x', string(MicNum), 'x', string(points_rir), '.mat'];
% rir_filemane = join(rir_filename_str, '');
% save(rir_filemane, 'h')

% % load RIR 的 .mat 檔 %
% rir_filename_str = ['h\h_', string(reverberation_time), 'x', string(MicNum), 'x', string(points_rir), '.mat'];
% rir_filemane = join(rir_filename_str, '');
% load(rir_filemane)

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

%% RIR mix source (x) %%
% convolution source and RIR %
as = zeros(MicNum, points_rir+SorLen-1);
for i = 1 : MicNum
    as(i, :) = conv(h(i, :), source);
end

x = as(:, 1:SorLen);

%% MCN %%
% basic parameters %
L = points_rir;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = 0.999;
step_size = 0.999;
diaload_delta_h = 0;
stop_criteria = 1e-8;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize R %
R_ini_scalar = 0;
for i = 1:MicNum
    R_ini_scalar = R_ini_scalar + flip(x(i, 1:L))*flip(x(i, 1:L)).'/L;
end

R = R_ini_scalar*eye(MicNum*L);

% initialize h_hat %
h_hat = zeros(MicNum*L, 1);
for i = 1:MicNum
    h_hat((i-1)*L+1) = 1;
end

h_hat = h_hat/norm(h_hat);

% initialize cost function %
cost_fun = zeros(SorLen, 1);

% iteration process %
for n = L+1:SorLen
    % construct R %
    R_temp = zeros(MicNum*L, MicNum*L);
    dia_sum = zeros(L, L);
    for i = 1:MicNum
        for j = 1:MicNum
            R_temp((i-1)*L+1:i*L, (j-1)*L+1:j*L) = -(flip(x(j, n-L+1:n)).'*flip(x(i, n-L+1:n)));
            if i == j
                dia_sum = dia_sum + flip(x(j, n-L+1:n)).'*flip(x(i, n-L+1:n));
            end

        end

    end

    for i = 1:MicNum
        R_temp((i-1)*L+1:i*L, (i-1)*L+1:i*L) = R_temp((i-1)*L+1:i*L, (i-1)*L+1:i*L) + dia_sum;
    end

    R = lambda*R + R_temp;

    % construct cost function %
    for i = 1:MicNum-1
        for j = i+1:MicNum
            cost_fun(n, :) = cost_fun(n, :) + (flip(x(i, n-L+1:n))*h_hat((j-1)*L+1:j*L, :) - flip(x(j, n-L+1:n))*h_hat((i-1)*L+1:i*L, :))^2;
        end

    end

    % compute new h_hat %
    delta_h = inv(R-2*h_hat*(h_hat.')*R-2*R*h_hat*(h_hat.')+diaload_delta_h*eye(MicNum*L))*(R_temp*h_hat-cost_fun(n, :)*h_hat);    % diaload_delta_h*eye(MicNum*L)
    h_hat = (h_hat-step_size*delta_h)/norm(h_hat-step_size*delta_h);

    % stop criteria %
    cost_mean = mean(cost_fun(n-9:n, :));
    if cost_mean < stop_criteria
        break;
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 中途看波型跟NRMSPM %
aa = reshape(h_hat, [size(h_hat, 1)/MicNum MicNum]).';
ratio_aa = zeros(MicNum, 1);
for i = 1:MicNum
    ratio_aa(i, :) = max(abs(h(i, :)))/max(abs(aa(i, :)));
end

aa = aa.*ratio_aa;


figure(3)
plot(h(look_mic, :), 'r');
hold on
plot(-aa(look_mic, :), 'b');
hold off
xlim([1 points_rir])
legend('ground-truth RIR', 'estimated RIR')
xlabel('time samples')
ylabel('amplitude')
shg

h_NRMSPM = reshape(h.', [MicNum*L 1]);
aa_NRMSPM = reshape(aa.', [MicNum*L 1]);
NRMSPM = 20*log10(norm(h_NRMSPM-h_NRMSPM.'*aa_NRMSPM/(aa_NRMSPM.'*aa_NRMSPM)*aa_NRMSPM)/norm(h_NRMSPM));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 畫圖看結果 %%
% reshape and rescale h_hat %
h_hat = reshape(h_hat, [size(h_hat, 1)/MicNum MicNum]).';
ratio_h_hat = zeros(MicNum, 1);
for i = 1:MicNum
    ratio_h_hat(i, :) = max(abs(h(i, :)))/max(abs(h_hat(i, :)));
end

h_hat = h_hat.*ratio_h_hat;

A_filename_str = ['A_tdomain\BSI_NA_TD_', string(reverberation_time), '_h_hat.mat'];
A_filename = join(A_filename_str, '');
save(A_filename, 'h_hat')

% cost function 圖 %
figure(3)
plot(cost_fun(L+1:end, :).');
xlabel('update times')
title('cost function')

fig_filename_str = ['fig\BSI_NA_TD_', string(reverberation_time), '_cost_fun.fig'];
fig_filename = join(fig_filename_str, '');
savefig(fig_filename)

% RIR 比較圖 %
figure(4)
plot(h(look_mic, :), 'r');
hold on
plot(-h_hat(look_mic, :), 'b');
hold off
xlim([1 points_rir])
legend('tfestimate', 'BSI')
title('RIR')
xlabel('time samples')
ylabel('amplitude')
shg

fig_filename_str = ['fig\BSI_NA_TD_', string(reverberation_time), '_RIR.fig'];
fig_filename = join(fig_filename_str, '');
savefig(fig_filename)

%% NRMSPM %%
h_NRMSPM = reshape(h.', [MicNum*points_rir 1]);
h_hat_NRMSPM = reshape(h_hat.', [MicNum*points_rir 1]);
NRMSPM = 20*log10(norm(h_NRMSPM-h_NRMSPM.'*h_hat_NRMSPM/(h_hat_NRMSPM.'*h_hat_NRMSPM)*h_hat_NRMSPM)/norm(h_NRMSPM));

NRMSPM_in = zeros(MicNum, 1);
for i = 1:MicNum
    NRMSPM_in(i, :) = 20*log10(norm(h(i, :).'-h(i, :)*h_hat(i, :).'/(h_hat(i, :)*h_hat(i, :).')*h_hat(i, :).')/norm(h(i, :).'));
end


%% 檢查 direct sound 有無 match %%
[~, argmax_h] = max(abs(h.'));
[~, argmax_h_hat] = max(abs(h_hat.'));

%% ATF magnitude and phase plot %%
ATF = fft(h, points_rir, 2);
ATF_estimated = fft(h_hat, points_rir, 2);

figure(5)
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

fig_filename_str = ['fig\BSI_NA_TD_', string(reverberation_time), '_ATF.fig'];
fig_filename = join(fig_filename_str, '');
savefig(fig_filename)

fprintf('done\n')
