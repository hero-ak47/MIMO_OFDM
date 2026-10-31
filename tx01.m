%% =====================================================================
%  MIMO-OFDM TRANSMITTER 2Tx - SFBC (Alamouti tren tan so), streaming
%
%   - BPSK, 32 active SC: k = -16..-1, +1..+16
%   - Nhom 4 SC: [P1 P2 D D] lap 8 lan
%         Tx1: p   0   s1      s2
%         Tx2: 0   p  -conj(s2) conj(s1)
%     -> pilot truc giao theo tan so (moi Tx 8 pilot), 8 cap SFBC
%        = 16 bit / OFDM symbol
%   - Khung: guard -> chirp (CHI Tx1) -> gap -> DATA symbols -> guard
%   - Khong con training symbol: kenh duoc uoc luong tu pilot cua
%     CHINH symbol do (kenh doi theo tung symbol)
%   - Fs_bb = 8 kHz, Fs_dac = 48 kHz, fc = 14 kHz
%   - Dau ra: wav STEREO (cot 1 = Tx1 / loa trai, cot 2 = Tx2 / loa phai)
%% =====================================================================

clear; clc; close all;

%% ==================== THAM SO HE THONG ==============================

Fs_bb   = 8000;
Fs_dac  = 48000;
Lup     = Fs_dac/Fs_bb;      % 6

N       = 256;
CP      = 96;

fc      = 14000;
df      = Fs_bb / N;         % 31.25 Hz

nTx     = 2;

% Che do PHAT (khong anh huong file wav tham chieu):
%   'mimo' = 2 loa | 'tx1' = chi loa 1 | 'tx2' = chi loa 2 (de thu rieng tung link)
txMode  = 'mimo';

%% ==================== PHAN BO SUBCARRIER ============================

kUsed = [-16:-1, 1:16];                  % 32 SC, bo DC
nSC   = numel(kUsed);
bin   = mod(kUsed, N) + 1;               % bin MATLAB (1-based)

g = 0:7;
pilotPos = [4*g+1; 4*g+2].';             % 8x2: cot t = vi tri pilot cua Tx t
dataPos  = reshape([4*g+3; 4*g+4], 1, []);   % 1x16, tung cap ke nhau
k1i = dataPos(1:2:end);                  % SC dau cua cap
k2i = dataPos(2:2:end);                  % SC sau cua cap

nP = size(pilotPos,1);                   % 8 pilot / Tx
nD = numel(dataPos);                     % 16 data SC
bitsPerSym = nD;                         % 16 bit / symbol (BPSK)

% Pilot BPSK co dinh (giam PAPR so voi toan +1), moi Tx 1 day
pilotVal = [ 1  1;
            -1  1;
             1 -1;
             1  1;
            -1 -1;
             1 -1;
            -1  1;
            -1 -1];

% Cong suat: tong cong suat 2 Tx = cong suat SISO
%   data: bien do 1/sqrt(2) moi Tx
%   pilot: bien do 1 (chi 1 Tx phat tai SC pilot) -> pilot "noi" len
%   Moi Tx: 8*1^2 + 16*0.5 = 16 ; tong = 32 = nSC (nhu SISO)
dataAmp  = 1/sqrt(nTx);
pilotAmp = 1;

%% ==================== BAN TIN =======================================

msg = 'Ra di mang nang loi the, chua thang giac My chua ve Bach Khoa.';

fprintf('\n============================================================\n');
fprintf('BAN TIN\n');
fprintf('============================================================\n');
fprintf('"%s"\n', msg);

%% ==================== TEXT -> BIT ===================================

bits    = reshape(dec2bin(double(msg), 8).' - '0', [], 1);
numBits = numel(bits);

nSym   = ceil(numBits / bitsPerSym);
numPad = nSym * bitsPerSym - numBits;

txBits = [bits; zeros(numPad, 1)];

%% ==================== PREAMBLE CHIRP ================================

chirp_dur = 0.05;
t_c = (0:round(chirp_dur*Fs_bb)-1)' / Fs_bb;

f0 = -800;
f1 =  800;

c_phase = 2*pi * (f0*t_c + (f1-f0)/(2*chirp_dur) * t_c.^2);

syncSym = exp(1j*c_phase);
syncSym = syncSym .* hann(numel(syncSym));

%% ==================== DATA SYMBOLS (SFBC) ===========================

dBits     = reshape(txBits, bitsPerSym, nSym);
symbolLen = N + CP;
dataSyms  = zeros(symbolLen * nSym, nTx);   % cot 1 = Tx1, cot 2 = Tx2

for s = 1:nSym

    d  = 1 - 2*dBits(:, s);              % BPSK: 0 -> +1, 1 -> -1
    s1 = d(1:2:end);                     % 8 symbol "dau cap"
    s2 = d(2:2:end);                     % 8 symbol "sau cap"

    X1 = zeros(N,1);
    X2 = zeros(N,1);

    % pilot truc giao (FDM)
    X1(bin(pilotPos(:,1))) = pilotAmp * pilotVal(:,1);
    X2(bin(pilotPos(:,2))) = pilotAmp * pilotVal(:,2);

    % SFBC:  SC k1 | SC k2
    %   Tx1:  s1   |  s2
    %   Tx2: -s2*  |  s1*
    X1(bin(k1i)) =  dataAmp * s1;
    X1(bin(k2i)) =  dataAmp * s2;
    X2(bin(k1i)) = -dataAmp * conj(s2);
    X2(bin(k2i)) =  dataAmp * conj(s1);

    idx = (s-1)*symbolLen + (1:symbolLen);
    dataSyms(idx,1) = ofdmMod(X1, N, CP);
    dataSyms(idx,2) = ofdmMod(X2, N, CP);

    if s == 1
        X_first = [X1, X2];
    end
end

fprintf('\nSo OFDM data symbol = %d  (%d bit/symbol)\n', nSym, bitsPerSym);
fprintf('Symbol #1, SC active (k, Tx1, Tx2):\n');
disp([kUsed(:), X_first(bin,1), X_first(bin,2)]);

%% ==================== GHEP KHUNG ====================================

guardLen = round(Fs_bb * 0.2);       % 0.2 s
gapLen   = round(Fs_bb * 0.02);      % 20 ms

zg = zeros(guardLen, 1);
zp = zeros(gapLen,   1);

% Tx1: co chirp.  Tx2: im lang trong luc chirp
tx_bb = [
    [zg,       zg       ];     % guard
    [syncSym,  0*syncSym];     % chirp: Tx1 phat, Tx2 im
    [zp,       zp       ];     % gap
    dataSyms;                  % N x 2 (cot 1 = Tx1, cot 2 = Tx2)
    [zg,       zg       ]      % guard
];

%% ==================== UPSAMPLE ======================================

Nfilt    = 96;
h_interp = Lup * fir1(Nfilt - 1, 1/Lup);

tx_up = zeros(size(tx_bb,1)*Lup, nTx);
tx_up(1:Lup:end, :) = tx_bb;
tx_up = filter(h_interp, 1, tx_up);          % loc tung cot

%% ==================== UPCONVERSION ==================================

n       = (0:size(tx_up,1)-1).';
carrier = exp(1j * 2*pi*fc/Fs_dac*n);
tx_pass = real(tx_up .* carrier);            % N x 2

% Chuan hoa CHUNG 1 he so cho ca 2 kenh (giu can bang cong suat Tx1/Tx2)
tx_pass = 0.8 * tx_pass / max(abs(tx_pass(:)));

%% ==================== LUU FILE ======================================

wavFile = fullfile(pwd, 'ofdm_tx_signal.wav');
audiowrite(wavFile, tx_pass, Fs_dac, 'BitsPerSample', 24);

refFile = fullfile(pwd, 'ofdm_tx_ref.mat');
save(refFile, ...
    'Fs_bb', 'Fs_dac', 'Lup', 'N', 'CP', 'fc', 'df', 'nTx', ...
    'kUsed', 'nSC', 'bin', 'pilotPos', 'dataPos', ...
    'nP', 'nD', 'pilotVal', 'pilotAmp', 'dataAmp', 'bitsPerSym', ...
    'syncSym', 'gapLen', ...
    'txBits', 'bits', 'msg', 'nSym', ...
    'h_interp', 'Nfilt');

%% ==================== PHAT QUA LOA ==================================

fprintf('\n============================================================\n');
fprintf('NHAN ENTER DE PHAT QUA LOA (STEREO: L = Tx1, R = Tx2)\n');
fprintf('============================================================\n');

pause;

tx_play = tx_pass;
switch txMode
    case 'tx1', tx_play(:,2) = 0;
    case 'tx2', tx_play(:,1) = 0;
end
fprintf('Dang phat... (%.2f s) [txMode = %s]\n', size(tx_pass,1)/Fs_dac, txMode);
sound(tx_play, Fs_dac);
pause(size(tx_pass,1)/Fs_dac + 0.5);
fprintf('Phat xong!\n');

%% =====================================================================
%                           HAM PHU
%% =====================================================================

function s = ofdmMod(X, N, CP)
    x = ifft(X) * sqrt(N);
    s = [x(end-CP+1:end); x];
end
