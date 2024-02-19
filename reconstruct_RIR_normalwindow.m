function atf = reconstruct_RIR_normalwindow(points_rir, NFFT, hopsize, L, win, fs, frequency, MicNum, CTF)
% CTF dimension : frequency x L x MicNum

ATF_frame = (points_rir - NFFT)/hopsize + 1 + (round(NFFT/hopsize)-1);    % 使 ATF 長度等於 points_rir + (osfac-1)*hopsize (因為回時域後還許把前 (osfac-1)*hopsize 點去掉)
impulse = zeros(1, ((ATF_frame+L-1)-1)*hopsize+NFFT);    % impulse signal 在頻域的的長度需等於 ATF_frame + L - 1 (因為 ?-L+1=ATF_frame ?=ATF_frame+L-1)
impulse(:, (L-1)*hopsize + 1) = 1;    % 因為是 convolution 所以 impulse 必須在 (L-1)*hopsize + 1 等等 flip 來才會是第一個 frame
impulse_transpose = impulse.';
[IMPULSE, ~, ~] = stft(impulse_transpose, fs, Window=win, OverlapLength=NFFT-hopsize, FFTLength=NFFT, FrequencyRange='onesided');

ATF = zeros(frequency, ATF_frame, MicNum);
for FrameNo= L:ATF_frame+L-1
    for i = 1 : MicNum
        ATF(:, FrameNo-L+1, i) = sum(CTF(:, :, i).*flip(IMPULSE(:, FrameNo-L+1:FrameNo), 2), 2);
    end
end

[atf_transpose] = istft(ATF, fs, Window=win, OverlapLength=NFFT-hopsize, FFTLength=NFFT, ConjugateSymmetric=true, FrequencyRange='onesided');
atf = atf_transpose.';

% output 須把前 (osfac-1)*hopsize 點去掉