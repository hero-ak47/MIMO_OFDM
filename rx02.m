%% =====================================================================
%  MIMO-OFDM RECEIVER 2Tx - SFBC (Streaming, Acoustic Robust)   [v3: estMode ls|rlsFixed|rlsVFF|kalmanAR1|movAvg]
%
%   - nRx = 1 hoac 2 mic: downconvert 14 kHz -> baseband, LPF, decimate 6
%   - Chirp matched filter (cong nang luong cac mic) -> sync
%   - Moi OFDM symbol: FFT -> uoc luong H(k,t,r) tu pilot truc giao
%     cua CHINH symbol do -> khu do doc pha -> noi suy -> MMSE-SFBC
%   - txActive = [1 1] (MIMO) hoac [1 0] (chi Tx1 phat -> SISO / 1x2)
%   - mode: 'record' / 'loopback' / 'sim' / 'file' / 'array'
%       'array' + batchMode = true: dung de chay Monte Carlo tu
%       ofdm_fading_compare.m (tin hieu truyen qua bien bt_y)
%% =====================================================================

if ~exist('batchMode','var') || ~batchMode
    clear; clc; close all;
    batchMode = false;
end

%% ==================== CHE DO HOAT DONG ==============================

if ~batchMode
    mode     = 'sim';       % 'record' / 'loopback' / 'sim' / 'file'
    nRx      = 2;           % so mic (1 hoac 2)
    txActive = [1 1];       % Tx nao dang phat that su
    sync_thr = 0.3;
else
    mode     = 'array';
    nRx      = bt_nRx;
    txActive = bt_txActive;
    sync_thr = bt_syncThr;
end

recDuration = 10;
rxWavFile   = 'ofdm_rx_mimo_recorded.wav';

% --- tham so mode 'sim' (kenh THUC o passband, hoi bien thien) ---
sim_snr       = 15;                 % dB, rong bang tai mic
sim_delays_ms = [0 1.5 4 7];        % ms (max-min phai < 8 ms voi pilot cach 4 SC)
sim_gains     = [1 .6 .4 .2];
sim_spk_ms    = 0.9;                % lech tre them ngau nhien moi link
sim_rho       = 0.8;                % tuong quan kenh dau/cuoi frame

% Lui diem bat dau FFT ve truoc (nam trong CP) de tranh ISI.
symBackoff = 8;        % don vi: mau baseband (8 kHz)

%% ==================== UOC LUONG KENH (v3, muc 7-8 handoff) ===========
%  est.mode:
%    'ls'        LS + khu doc pha + noi suy (ban v2, khong nho trang thai)
%    'rlsFixed'  RLS vo huong, lambda co dinh (est.lamFixed)
%    'rlsVFF'    RLS lambda bien thien theo NIS, 1 lambda / link (muc 7)
%    'kalmanAR1' Kalman AR(1): rho = est.rho, hoac J0(2*pi*est.fd*Tsym)
%    'movAvg'    trung binh truot est.win symbol (da bu pha)
%  est.comp = true: bu CPE + do doc pha moi symbol (so voi trang thai) truoc
%  khi vao bo loc, roi gan lai khi xuat H (Buoc 4). Bo qua voi mode 'ls'.
est = struct('mode','ls', 'comp',true, ...
             'lamFixed',0.7, 'lamMin',0.3, 'lamMax',0.9, ...
             'alpha',0.6, 'beta',1.0, 'nisVar','Q', ...   % nisVar: 'Q' (handoff) | 'V' (chinh xac cho kenh tinh)
             'rho',0.84, 'fd',[], 'win',4);

% Batch: bt_est (struct) ghi de tung truong; bt_estMode la cach viet tat.
if batchMode
    if exist('bt_est','var') && isstruct(bt_est)
        fn = fieldnames(bt_est);
        for ii = 1:numel(fn), est.(fn{ii}) = bt_est.(fn{ii}); end
    end
    if exist('bt_estMode','var'), est.mode = bt_estMode; end
end
if ~ismember(est.mode, {'ls','rlsFixed','rlsVFF','kalmanAR1','movAvg'})
    error('est.mode khong hop le: %s', est.mode);
end

if ~batchMode
    fprintf('=== MIMO-OFDM Receiver SFBC [mode: %s, nRx = %d, txActive = %s] ===\n', ...
            mode, nRx, mat2str(txActive));
end

%% ==================== LOAD THAM SO TU TX =============================

refFile = 'ofdm_tx_ref.mat';
if ~exist(refFile, 'file')
    error(['Khong tim thay ', refFile, '! Hay chay ofdm_tx_mimo truoc.']);
end
load(refFile);

if nTx ~= 2
    error('Code nay danh cho TX 2 anten.');
end
if nRx < 1 || nRx > 2
    error('nRx phai la 1 hoac 2.');
end

k1i   = dataPos(1:2:end);
k2i   = dataPos(2:2:end);
nPair = numel(k1i);
if any(k2i - k1i ~= 1)
    error('Cap SFBC phai nam ke nhau.');
end
if bitsPerSym ~= nD
    error('bitsPerSym khong khop voi nD!');
end

SL       = N + CP;
chScale  = dataAmp / pilotAmp;       % pilot -> kenh hieu dung cua data

% Bin dung de uoc luong nhieu (ngoai bang tan tin hieu, trong bang LPF)
noiseBin = mod([-80:-48, 48:80], N) + 1;

%% ==================== CHUAN BI SYNC =================================

L_sync = numel(syncSym);
E_sync = sum(abs(syncSym).^2);

%% ==================== NGUON TIN HIEU ================================

isRealtime    = strcmp(mode, 'record');
chunkSize_dac = 1024;

if ~isRealtime
    switch mode
        case 'loopback'
            [y_pass_all, ~] = audioread('ofdm_tx_signal.wav');
            if nRx == 1
                y_pass_all = sum(y_pass_all, 2);     % 1 mic nghe ca 2 loa
            end

        case 'sim'
            [x2, ~] = audioread('ofdm_tx_signal.wav');     % M x 2
            rng(42);
            M0 = size(x2,1);
            w  = (0:M0-1).' / (M0-1);
            y_pass_all = zeros(M0, nRx);

            for r = 1:nRx
                for t = 1:nTx
                    extra = rand * sim_spk_ms;
                    idx   = round((sim_delays_ms + extra)*1e-3*Fs_dac) + 1;
                    h1 = simTaps(idx, sim_gains);
                    hn = simTaps(idx, sim_gains);
                    h2 = sim_rho*h1 + sqrt(1-sim_rho^2)*hn;
                    y_pass_all(:,r) = y_pass_all(:,r) ...
                        + (1-w).*filter(h1, 1, x2(:,t)) ...
                        +    w .*filter(h2, 1, x2(:,t));
                end
                sigP   = mean(y_pass_all(:,r).^2);
                noiseP = sigP / 10^(sim_snr/10);
                y_pass_all(:,r) = y_pass_all(:,r) + sqrt(noiseP)*randn(M0,1);
            end

        case 'file'
            [y_pass_all, ~] = audioread(rxWavFile);

        case 'array'
            y_pass_all = bt_y;

        otherwise
            error('Mode khong hop le.');
    end

    if size(y_pass_all,2) < nRx
        error('Nguon tin hieu phai co it nhat %d kenh.', nRx);
    end
    y_pass_all = y_pass_all(:, 1:nRx);
    passPtr = 1;
end

%% ==================== FRONT-END =====================================

nco_phase = 0;
nco_freq  = 2*pi*fc/Fs_dac;

h_aa  = 2 * fir1(Nfilt-1, 1/Lup);
zf_aa = zeros(Nfilt-1, nRx);

decim_phase = 0;

%% ==================== RECEIVER STATE ================================

state       = 0;
buf         = complex(zeros(0, nRx));
rp          = 1;
syncFound   = false;
nSymDecoded = 0;

rxBits = false(bitsPerSym, nSym);
eqAll  = complex(zeros(bitsPerSym, nSym));
Hhist  = complex(zeros(nSC, nTx, nRx, nSym));
snrEst = zeros(1, nSym);
nvS    = [];

% Trang thai bo uoc luong theo tung link (t,r)
Tsym = SL / Fs_bb;                                  % 44 ms
if ~isempty(est.fd)
    est.rho = besselj(0, 2*pi*est.fd*Tsym);         % tuong quan Jakes giua 2 symbol
end
stLink = cell(nTx, nRx);
for t = 1:nTx
    for r = 1:nRx
        stLink{t,r} = struct('Hs',[], 'Q',0, 'V',0, 'P',0, 'zbuf',[], 'init',false);
    end
end
lamHist = nan(nTx, nRx, nSym);                      % lambda (hoac 1-K) theo symbol

if ~batchMode
    fprintf('Uoc luong kenh: %s | comp = %d', est.mode, est.comp);
    switch est.mode
        case 'rlsFixed',  fprintf(' | lambda = %.2f', est.lamFixed);
        case 'rlsVFF',    fprintf(' | lam [%.2f, %.2f] alpha=%.2f beta=%.2f nisVar=%s', ...
                                  est.lamMin, est.lamMax, est.alpha, est.beta, est.nisVar);
        case 'kalmanAR1', fprintf(' | rho = %.3f', est.rho);
        case 'movAvg',    fprintf(' | win = %d', est.win);
    end
    fprintf('\n');
end

%% ==================== CHIRP MATCHED FILTER ==========================

h_match  = flipud(conj(syncSym));
zf_match = zeros(numel(h_match)-1, nRx);

h_energy  = ones(L_sync,1);
zf_energy = zeros(numel(h_energy)-1, nRx);

metric_buf = [];
totalBB = 0;  candM = 0;  candAbs = 0;

%% ==================== REALTIME AUDIO ================================

if isRealtime
    hasAudioTbx = ~isempty(ver('audio'));

    if hasAudioTbx
        reader = audioDeviceReader('SampleRate', Fs_dac, ...
                                   'SamplesPerFrame', chunkSize_dac, ...
                                   'NumChannels', nRx);
        cleanupObj = onCleanup(@() release(reader));
    else
        rec = audiorecorder(Fs_dac, 16, nRx);
        record(rec);
        lastReadIdx = 0;
        cleanupObj = onCleanup(@() stop(rec));
    end

    fprintf('\n>>> DANG THU (%ds)...\n', recDuration);
    tic;
end

%% ==================== MAIN LOOP =====================================

finished   = false;
y_pass_log = [];

while ~finished

    % ---------------- DOC CHUNK (nSamp x nRx) ----------------
    if isRealtime
        if hasAudioTbx
            [audioChunk, nOver] = reader();
            if nOver > 0
                fprintf('OVERRUN: %d mau\n', nOver);
            end
        else
            pause(chunkSize_dac/Fs_dac*0.8);
            allData = getaudiodata(rec, 'double');
            nTotal  = size(allData,1);
            if nTotal <= lastReadIdx
                continue;
            end
            audioChunk  = allData(lastReadIdx+1:nTotal, :);
            lastReadIdx = nTotal;
        end

        if toc > recDuration
            finished = true;
        end
    else
        if passPtr > size(y_pass_all,1)
            finished = true;
            continue;
        end
        iEnd       = min(passPtr + chunkSize_dac - 1, size(y_pass_all,1));
        audioChunk = y_pass_all(passPtr:iEnd, :);
        passPtr    = iEnd + 1;
    end

    nSamp = size(audioChunk,1);
    if nSamp == 0
        continue;
    end

    if isRealtime
        y_pass_log = [y_pass_log; audioChunk]; %#ok<AGROW>
    end

    % ---------------- DOWNCONVERT ----------------
    n_vec = (0:nSamp-1).';
    y_iq  = audioChunk .* exp(-1j*(nco_phase + nco_freq*n_vec));
    nco_phase = mod(nco_phase + nco_freq*nSamp, 2*pi);

    % ---------------- LPF ----------------
    [y_filt, zf_aa] = filter(h_aa, 1, y_iq, zf_aa);

    % ---------------- DECIMATE 48k -> 8k ----------------
    firstOut = mod(Lup-decim_phase, Lup) + 1;

    if firstOut > nSamp
        bb_chunk = complex(zeros(0, nRx));
    else
        bb_chunk = y_filt(firstOut:Lup:nSamp, :);
    end
    decim_phase = mod(decim_phase + nSamp, Lup);

    if isempty(bb_chunk)
        continue;
    end

    buf     = [buf; bb_chunk]; %#ok<AGROW>
    totalBB = totalBB + size(bb_chunk,1);

    % =================================================================
    % STATE 0: CHIRP SYNC (cong nang luong cac mic, chot dinh sau khi
    % da qua dinh >= 12 mau de khong tuot dinh nam o bien chunk)
    % =================================================================
    if state == 0

        [corr_out, zf_match]    = filter(h_match, 1, bb_chunk, zf_match);
        [energy_out, zf_energy] = filter(h_energy, 1, abs(bb_chunk).^2, zf_energy);

        M = sum(abs(corr_out).^2, 2) ./ (sum(energy_out, 2)*E_sync + 1e-6);
        metric_buf = [metric_buf; M]; %#ok<AGROW>

        [maxM, mIdx] = max(M);
        if maxM > sync_thr && maxM > candM
            candM   = maxM;
            candAbs = totalBB - size(bb_chunk,1) + mIdx;    % toa do tuyet doi
        end

        if candM > 0 && totalBB - candAbs > 12
            idx_in_buf = candAbs - (totalBB - size(buf,1)); % vi tri ket thuc chirp trong buf
            rp = max(idx_in_buf + gapLen + 1 - symBackoff, 1);

            state     = 1;
            syncFound = true;

            if ~batchMode
                fprintf('[SYNC] Preamble found, M = %.3f\n', candM);
            end
        end

        if state == 0 && size(buf,1) > 3*SL
            buf(1:size(buf,1)-2*SL, :) = [];
        end
    end

    % =================================================================
    % STATE 1: MIMO-OFDM DEMODULATION
    % =================================================================
    while state == 1 && nSymDecoded < nSym

        needEnd = rp + SL - 1;
        if needEnd > size(buf,1)
            break;
        end

        seg = buf(rp:needEnd, :);                       % SL x nRx
        Yf  = fft(seg(CP+1:end, :)) / sqrt(N);          % N x nRx
        Y   = Yf(bin, :);                               % nSC x nRx

        % --- Uoc luong nhieu (lam muot theo symbol) ---
        nv   = mean(abs(Yf(noiseBin,:)).^2, 1);         % 1 x nRx
        sigP = mean(abs(Y).^2, 1);
        nv   = max(nv, 1e-4*sigP);
        if isempty(nvS)
            nvS = nv;
        else
            nvS = 0.9*nvS + 0.1*nv;
        end

        % --- Uoc luong kenh MOI SYMBOL tu pilot truc giao ---
        H = zeros(nSC, nTx, nRx);
        nSymDecoded = nSymDecoded + 1;
        for r = 1:nRx
            sz2 = nvS(r) * chScale^2;                   % phuong sai nhieu cua z
            for t = 1:nTx
                if ~txActive(t), continue; end          % Tx khong phat: khong cap nhat
                pp = pilotPos(:,t);
                z  = Y(pp, r) ./ pilotVal(:,t) * chScale;      % LS tai pilot (|X| = 1)
                [H(:,t,r), stLink{t,r}, lamHist(t,r,nSymDecoded)] = ...
                    trackLink(z, kUsed(pp), kUsed, stLink{t,r}, sz2, est);
            end
        end
        H(:, txActive == 0, :) = 0;       % Tx khong phat -> bo kenh (tranh nhieu)

        Hhist(:,:,:,nSymDecoded) = H;
        snrEst(nSymDecoded) = 10*log10(mean(sigP ./ nvS));

        % --- Tach SFBC: MMSE tong quat cho tung cap ---
        for p = 1:nPair
            a1 = k1i(p);
            a2 = k2i(p);

            A  = zeros(2*nRx, 2);
            yy = zeros(2*nRx, 1);
            wv = zeros(2*nRx, 1);

            for r = 1:nRx
                ii = 2*r-1:2*r;
                A(ii,:) = [ H(a1,1,r),        -H(a1,2,r);
                            conj(H(a2,2,r)),   conj(H(a2,1,r)) ];
                yy(ii)  = [ Y(a1,r); conj(Y(a2,r)) ];
                wv(ii)  = 1 / nvS(r);
            end

            Aw = A .* wv;
            Mx = Aw'*A + eye(2);
            z  = Mx \ (Aw'*yy);                % z = [s1 ; conj(s2)]
            G  = Mx \ (Aw'*A);
            zu = z ./ max(real(diag(G)), 1e-6);

            rxBits(2*p-1, nSymDecoded) = real(zu(1)) < 0;
            rxBits(2*p  , nSymDecoded) = real(zu(2)) < 0;

            eqAll(2*p-1, nSymDecoded) = zu(1);
            eqAll(2*p  , nSymDecoded) = conj(zu(2));
        end

        % --- Symbol ke tiep ---
        rp   = rp + SL;
        drop = rp - 10;
        if drop > 0
            buf(1:drop, :) = [];
            rp = rp - drop;
        end
    end

    if state == 1 && nSymDecoded >= nSym
        if ~batchMode
            fprintf('\n>>> DA GIAI MA XONG!\n');
        end
        finished = true;
    end
end

%% ====================================================================
%                          KET QUA
%% ====================================================================

if ~syncFound
    error('LOI: Khong tim thay Preamble!');
end

rb = rxBits(:);

numMsgBits = length(bits);

if length(rb) < numMsgBits
    rb_msg  = rb;
    ref_msg = bits(1:length(rb));
else
    rb_msg  = rb(1:numMsgBits);
    ref_msg = bits;
end

ber = mean(rb_msg ~= ref_msg);

% --- NMSE kenh (Buoc 1): can bt_Htrue = H vat ly [nSC x nTx x nRx x nSym] ---
%  RX uoc luong kenh hieu dung cua data = dataAmp * H_vat_ly, nen nhan dataAmp.
%  nmseRaw : sai so tho (nhay voi lech pha/timing co dinh cua front-end)
%  nmseAl  : sau khi loai CPE + do doc pha toi uu tung symbol/link (do chat luong
%            ham theo SC, khong tinh bu tre co dinh). Don vi dB, chi tren SC data.
nmseRaw = NaN;  nmseAl = NaN;
if batchMode && exist('bt_Htrue','var') && nSymDecoded > 0
    nm = min(nSymDecoded, size(bt_Htrue,4));
    eR = 0;  eA = 0;  eT = 0;
    for m = 1:nm
        for r = 1:nRx
            for t = 1:nTx
                if ~txActive(t), continue; end
                Ht = dataAmp * bt_Htrue(:,t,r,m);
                He = Hhist(:,t,r,m);
                [cp, sl] = phaseFit(He .* conj(Ht), kUsed);
                He_al = He .* exp(-1j*(cp + sl*(kUsed(:) - mean(kUsed))));
                eR = eR + sum(abs(He(dataPos)    - Ht(dataPos)).^2);
                eA = eA + sum(abs(He_al(dataPos) - Ht(dataPos)).^2);
                eT = eT + sum(abs(Ht(dataPos)).^2);
            end
        end
    end
    nmseRaw = 10*log10(eR / eT);
    nmseAl  = 10*log10(eA / eT);
end

% SNR sau tach theo symbol (do lech cua diem chom sao so voi +-1)
nDec   = max(nSymDecoded, 1);
eqd    = eqAll(:, 1:nDec);
dd     = eqd - sign(real(eqd));
snrSym = -10*log10(mean(abs(dd).^2, 1) + 1e-12);
ss     = sort(snrSym);
snrP10 = ss(max(1, ceil(0.1*numel(ss))));

if ~batchMode
    fprintf('\n============================================================\n');
    fprintf('KET QUA\n');
    fprintf('============================================================\n');
    fprintf('So symbol giai ma = %d / %d\n', nSymDecoded, nSym);
    fprintf('So bit message    = %d\n', numMsgBits);
    fprintf('So bit RX         = %d\n', length(rb));
    fprintf('SNR/SC uoc luong  = %.1f dB (trung binh)\n', mean(snrEst(1:nDec)));
    fprintf('SNR sau tach      = TB %.1f | 10%% thap nhat %.1f | min %.1f dB\n', ...
            mean(snrSym), snrP10, min(snrSym));
    fprintf('BER               = %.6g\n', ber);
    if ~isnan(nmseRaw)
        fprintf('NMSE kenh (data)  = %.1f dB tho | %.1f dB sau can pha/doc\n', nmseRaw, nmseAl);
    end
    if any(strcmp(est.mode, {'rlsVFF','kalmanAR1'}))
        fprintf('lambda/(1-K) TB   = %.3f\n', mean(lamHist(:), 'omitnan'));
    end
end

%% ==================== BIT -> ASCII ==================================

if length(rb_msg) == numMsgBits
    rxByteBits = reshape(rb_msg, 8, []).';
    rxBytes    = bin2dec(char(rxByteBits + '0'));
    rxText     = char(rxBytes).';
else
    rxText = '';
end

if ~batchMode
    fprintf('\nBan tin thu:\n  "%s"\n', rxText);
end

%% ==================== LUU FILE THU ==================================

if isRealtime && ~isempty(y_pass_log)
    y_save = y_pass_log / max(abs(y_pass_log(:)) + eps);
    audiowrite(rxWavFile, y_save, Fs_dac, 'BitsPerSample', 24);
    fprintf('\nDa luu file thu: %s\n', rxWavFile);
end

%% ==================== HIEN THI ======================================

if ~batchMode

    figure('Name','MIMO SFBC RX','Position',[100 100 1300 700]);

    subplot(2,2,1);
    plot(metric_buf); grid on;
    title('Chirp Sync Metric'); xlabel('Samples'); ylabel('M');

    subplot(2,2,2);
    plot(real(eqAll(:)), imag(eqAll(:)), '.', 'MarkerSize', 6);
    hold on;
    plot([-1 1], [0 0], 'r+', 'MarkerSize', 12, 'LineWidth', 2);
    axis equal; grid on; xlim([-2 2]); ylim([-2 2]);
    title(sprintf('BPSK Constellation sau SFBC (BER = %.2g)', ber));
    xlabel('I'); ylabel('Q');

    subplot(2,2,3);
    if nSymDecoded > 0
        Hl = Hhist(:,:,:,nSymDecoded);
        mk = {'o-','s-','^-','d-'};
        lg = {};  cnt = 0;
        hold on;
        for rr = 1:nRx
            for tt = 1:nTx
                if txActive(tt)
                    cnt = cnt + 1;
                    plot(kUsed, abs(Hl(:,tt,rr)), mk{cnt});
                    lg{end+1} = sprintf('Tx%d->Rx%d', tt, rr); %#ok<SAGROW>
                end
            end
        end
        grid on; legend(lg, 'Location', 'best');
        title('|H(k)| cac link (symbol cuoi)');
        xlabel('Subcarrier k'); ylabel('|H|');
    end

    subplot(2,2,4);
    if nSymDecoded > 0
        mh = squeeze(mean(mean(abs(Hhist(:,1,:,1:nSymDecoded)),1),4));
        [~, rBest] = max(mh(:));
        imagesc(1:nSymDecoded, kUsed, ...
            20*log10(abs(squeeze(Hhist(:,1,rBest,1:nSymDecoded))) + 1e-6));
        axis xy; colorbar;
        title(sprintf('|H_{Tx1->Rx%d}| (dB) theo symbol', rBest));
        xlabel('Symbol'); ylabel('k');
    end
end

%% =====================================================================
%                           HAM PHU
%% =====================================================================

function h = simTaps(idx, gains)
    % Kenh THUC passband: cac tia Gaussian thuc, tre idx (mau 48 kHz)
    h = zeros(max(idx), 1);
    h(idx) = sqrt(gains(:)) .* randn(numel(idx), 1);
end

function [cpe, slope] = phaseFit(d, k)
    % Khop pha d(k) ~ cpe + slope*(k - mean(k)); do doc truoc (hieu pha cac cap
    % ke nhau, trong so |.|) de tranh quay vong, roi CPE tu tong da khu doc.
    d = d(:);  k = k(:);
    dd    = d(2:end) .* conj(d(1:end-1));
    w     = abs(dd) + 1e-12;
    slope = sum(w .* angle(dd) ./ diff(k)) / sum(w);
    cpe   = angle(sum(d .* exp(-1j*slope*(k - mean(k)))));
end

function [Hf, st, lam] = trackLink(z, kp, kAll, st, sz2, est)
    % Theo doi kenh 1 link (Tx t -> Rx r) tai cac pilot, roi noi suy ra moi SC.
    %   z   : LS tai pilot (nP x 1), da nhan chScale
    %   sz2 : phuong sai nhieu cua z
    %   st  : trang thai link (Hs, Q, V, P, zbuf, init)
    %   lam : lambda (RLS) hoac 1-K (Kalman) cua symbol nay, NaN neu khong co
    z = z(:);  kp = kp(:);  lam = NaN;

    if strcmp(est.mode, 'ls')
        Hf = estChan(z, kp, kAll);
        return;
    end

    % --- Buoc 4: bu CPE + do doc pha so voi trang thai truoc ---
    kc = mean(kp);
    cpe = 0;  slope = 0;
    if est.comp && st.init
        [cpe, slope] = phaseFit(z .* conj(st.Hs), kp);
    end
    zc = z .* exp(-1j*(cpe + slope*(kp - kc)));

    if ~st.init
        % Symbol dau: Q = 1, K = 1  ->  Hs = z (chinh la khoi tao LS)
        st.Hs = zc;  st.Q = 1;  st.V = 1;  st.P = sz2;
        st.zbuf = zc;  st.init = true;
    else
        switch est.mode
            case {'rlsFixed', 'rlsVFF'}
                e = zc - st.Hs;
                if strcmp(est.mode, 'rlsVFF')
                    if strcmp(est.nisVar, 'V'), den = 1 + st.V;
                    else,                       den = 1 + 1/max(st.Q, 1);
                    end
                    nis = mean(abs(e).^2) / (sz2 * den);   % ~1 neu kenh dung nhu du doan
                    u   = max(nis - 1, 0);
                    lam = est.lamMax - est.alpha * u / (u + est.beta);
                    lam = min(max(lam, est.lamMin), est.lamMax);
                else
                    lam = est.lamFixed;
                end
                st.Q  = lam*st.Q + 1;                      % Q = 1/P
                K     = 1 / st.Q;
                st.V  = (1-K)^2 * st.V + K^2;              % phuong sai Hs / sz2 (kenh tinh)
                st.Hs = st.Hs + K * e;

            case 'kalmanAR1'
                rho   = est.rho;
                sigH2 = max(mean(abs(zc).^2) - sz2, 1e-3*sz2);   % cong suat kenh / pilot
                xp    = rho * st.Hs;
                Pp    = rho^2 * st.P + (1 - rho^2) * sigH2;
                K     = Pp / (Pp + sz2);
                st.Hs = xp + K * (zc - xp);
                st.P  = (1 - K) * Pp;
                lam   = 1 - K;

            case 'movAvg'
                st.zbuf = [st.zbuf, zc];
                if size(st.zbuf, 2) > est.win
                    st.zbuf(:, 1) = [];
                end
                st.Hs = mean(st.zbuf, 2);
        end
    end

    % Noi suy theo tan so roi gan lai pha da bu
    Hf = estChan(st.Hs, kp, kAll) .* exp(1j*(cpe + slope*(kAll(:) - kc)));
end

function Hf = estChan(Hp, kp, kAll)
    % Hp : uoc luong kenh tai pilot cua 1 Tx / 1 Rx
    % kp : chi so k cua pilot (cach 4, hoac 5 khi qua DC)
    % 1) uoc luong do doc pha (tre trung tam), 2) khu, 3) noi suy, 4) tra lai
    Hp = Hp(:);  kp = kp(:);  kAll = kAll(:);

    d  = Hp(2:end) .* conj(Hp(1:end-1));
    dk = diff(kp);
    w  = abs(d) + 1e-12;
    slope = sum(w .* angle(d) ./ dk) / sum(w);        % rad / SC

    Hd = Hp .* exp(-1j*slope*kp);
    Hf = interp1(kp, Hd, kAll, 'linear', 'extrap') .* exp(1j*slope*kAll);
end
