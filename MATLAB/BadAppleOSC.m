clear,clc,close all
% Input a video and output a waveform file consisting of the coordinates of its edge points
% Left channel: horizontal coordinates
% Right channel: vertical coordinates

scanNumPF = 2; % Number of scans per frame
Fs = 48e3; % Sampling rate

[vidFile, vidPath] = uigetfile('*.avi;*.mpg;*.wmv;*.mp4;*.m4v;*.mov;*.mkv',...
'Select video file', '22118703_5_0.mp4');
[wavFile, wavPath] = uiputfile({'*.wav';'*.flac'}, 'Save audio file', 'PlayMe');

%% Read video file
disp('Loading file...');
Vid = VideoReader([vidPath vidFile]);
vidFrameRate = Vid.FrameRate; % Frame rate
nFrames = Vid.NumFrames; % Total number of frames
vidHeight = Vid.Height; % Height
vidWidth = Vid.Width; % Width
Vid.CurrentTime = 0; % Specify how many seconds from the beginning of the video to start reading
WHR = vidWidth/vidHeight;

dotNumPF = round(Fs/vidFrameRate); % Points per frame
dotNum = round(dotNumPF/scanNumPF); % Points per scan

%% Read frames and process
disp('Processing frames...');
Fig = waitbar(0,'Processing frames...');
bouDotxy = cell(nFrames, 1);
p0 = [(1024/WHR+1)/2, (1024+1)/2];
k = 1;
while hasFrame(Vid)
vidFrame = readFrame(Vid); % Read each frame
vidFrame = im2double(vidFrame);
vidFrame = rgb2gray(vidFrame);
vidFrame = imresize(vidFrame,[NaN 1024]);
vidFrame = imgaussfilt(vidFrame, 1024/dotNum); % Filter
% vidFrame = imbinarize(vidFrame); % Binarization
vidFrame = edge(double(vidFrame), 'Canny', [0.1 0.2]); % Edge detection
Bou = bwboundaries(vidFrame); % Get boundary coordinates

% Optimize order
[BouTemp,p0] = reorderlines(Bou,p0);
if isempty(Bou)
p0 = [(1024/WHR+1)/2, (1024+1)/2];
end

bouDot = cell2mat(BouTemp); % Each point on the boundary
bouDotNum = length(bouDot); % Number of points per frame
if bouDotNum > 0
bouDot = resample(bouDot, dotNum, bouDotNum, 0); % Adjust the number of points
bouDotTemp = repmat(bouDot, scanNumPF, 1); % Repeat the scanNumPF times per frame
else
bouDotTemp = NaN(dotNumPF, 2); % No picture
end

bouDotxy{k} = bouDotTemp; % Coordinates of all points to be drawn
waitbar(k/nFrames, Fig,...
sprintf('Processing frames...%.2f%%(%u/%u)',k/nFrames*100,k,nFrames));
k ​​= k + 1;
end
close(Fig)

%% Adjust the amplitude
disp('Adjust the amplitude...')
bouDotxy = cell2mat(bouDotxy);
bouDotxy = bouDotxy - mean(bouDotxy, 'omitnan'); % Remove DC
bouDotxy = bouDotxy / max(abs(bouDotxy),[],'all'); % Normalize
% Rotate 90° clockwise
bouDotxy(:,1) = -bouDotxy(:,1); % Flip horizontally
bouDotxy(:,[1 2]) = bouDotxy(:,[2 1]); % Swap xy
% Points without picture
bouDotxy(isnan(bouDotxy)) = 0;

%% Draw PSD
% Check the spectrum range, most of the energy should be in the listening range (20Hz~20kHz)
winlen = 2*Fs; % window length
window = hanning(winlen, 'periodic'); % window function
noverlap = winlen/2; % data overlap
nfft = winlen; % FFT points
[pxx, f] = pwelch(bouDotxy, window, noverlap, nfft, Fs, 'onesided');
semilogx(f, pxx)
xlabel('Frequency (Hz)')
ylabel('Power')

%% Output audio file
disp('Output...')
audiowrite([wavPath wavFile], bouDotxy, Fs)