clear,clc,close all
% ����һ����Ƶ,������ֵ�����Ե���������ɵĲ����ļ�
% ������:ˮƽ����
% ������:��ֱ����
scanNumPF = 2; % ÿ֡ɨ�����
Fs = 48e3; % ������
[vidFile, vidPath] = uigetfile('*.avi;*.mp4', 'ѡ����Ƶ�ļ�', '22118703_5_0.mp4');
[wavFile, wavPath] = uiputfile({'*.wav';'*.flac'}, '������Ƶ�ļ�', 'PlayMe');

%% ��ȡ��Ƶ�ļ�
disp('���ڼ����ļ�...');
Vid = VideoReader([vidPath vidFile]);
vidFrameRate = Vid.FrameRate; % ֡��
nFrames = Vid.NumFrames; % ��֡��
vidHeight = Vid.Height; % �߶�
vidWidth = Vid.Width; % ���
Vid.CurrentTime = 0; % ָ��Ӧ�ھ���Ƶ��ͷ�������λ�ÿ�ʼ��ȡ
WHR = vidWidth/vidHeight;

dotNumPF = Fs/vidFrameRate; % ÿ֡����
dotNum = dotNumPF/scanNumPF; % ÿ��ɨ�����

%% ��ȡ֡������
disp('���ڴ���֡...');
Fig = waitbar(0,'���ڴ���֡...');
bouDotxy = cell(dotNumPF*nFrames, 1);
p0 = [(1024/WHR+1)/2, (1024+1)/2];
k = 1;
while hasFrame(Vid)
    vidFrame = readFrame(Vid); % ��ȡÿ֡ͼ��
    vidFrame = im2double(vidFrame);
    vidFrame = rgb2gray(vidFrame);
    vidFrame = imresize(vidFrame,[NaN 1024]);
    vidFrame = imgaussfilt(vidFrame, 1024/dotNum) >= 0.5; % �˲�
    vidFrame = edge(double(vidFrame), 'Canny'); % ��Ե���
    Bou = bwboundaries(vidFrame); % ��ȡ�߽�����

    % �Ż�˳��
    [BouTemp,p0] = reorderlines(Bou,p0);
    if isempty(Bou)
        p0 = [(1024/WHR+1)/2, (1024+1)/2];
    end
    
    bouDot = cell2mat(BouTemp); % �߽��ϵ�ÿһ��
    bouDotNum = length(bouDot); % ÿһ֡�������
    if bouDotNum > 0
        bouDot = resample(bouDot, dotNum, bouDotNum, 0); % ��������
        bouDotTemp = repmat(bouDot, scanNumPF, 1); % ÿ֡�ظ�ɨ��scanNumPF��
    else
        bouDotTemp = NaN(dotNumPF, 2); % �޻���
    end

    bouDotxy{k} = bouDotTemp; % ����Ҫ��ĵ������
    waitbar(k/nFrames, Fig,...
        sprintf('���ڴ���֡...%.2f%%(%u/%u)',k/nFrames*100,k,nFrames));
    k = k + 1;
end
close(Fig)

%% ��������
disp('��������...')
bouDotxy = cell2mat(bouDotxy);
bouDotxy = bouDotxy - mean(bouDotxy, 'omitnan'); % �Ƴ�ֱ��
bouDotxy = bouDotxy / max(abs(bouDotxy),[],'all'); % ��һ��
% ˳ʱ����ת90��
bouDotxy(:,1) = -bouDotxy(:,1); % ˮƽ��ת
bouDotxy(:,[1 2]) = bouDotxy(:,[2 1]); % ����xy
% �޻���ĵ�
bouDotxy(isnan(bouDotxy)) = 0;

%% ����PSD
% �鿴Ƶ�׷�Χ,�󲿷�����Ӧ��������(20Hz~20kHz)
winlen = 2*Fs; % ������
window = hanning(winlen, 'periodic'); % ���ں���
noverlap = winlen/2; % �����ص�
nfft = winlen; % FFT����
[pxx, f] = pwelch(bouDotxy, window, noverlap, nfft, Fs, 'onesided');
semilogx(f, pxx)
xlabel('Ƶ��(Hz)')
ylabel('����')

%% �����Ƶ�ļ�
disp('���...')
audiowrite([wavPath wavFile], bouDotxy, Fs)

