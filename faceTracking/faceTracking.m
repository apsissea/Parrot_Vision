%% Params
rez = 0.5;
minThrsld = 30;
videoFileName = 'Bebop2_20180313082905+0100.mp4';

%% Objects
if ~exist('videoFReader','var') || ~exist('videoInf','var')
    videoFReader = vision.VideoFileReader(videoFileName,'VideoOutputDataType','uint8');
    videoInf = VideoReader(videoFileName);
    count = 1;
end

if exist('w','var')
    clearvars('w');
end

videoPlayer = vision.DeployableVideoPlayer;
peopleDetector = peopleDetectorACF;
nnet = alexnet;

seqLen = videoInf.NumberOfFrames;

%% Init
cont = ~isDone(videoFReader);
step(videoPlayer,imresize(step(videoFReader),rez));

% Configure waitbar to stay visible
w = waitbar(1,'1','Name','Video playing ...');
barPos = get(w,'position');
set(w,'position',[barPos(1) 100 barPos(3) barPos(4)]);

% int classifier
label = ' ';

%% Video loop

while cont
    frame = imresize(step(videoFReader),rez);
    
    % Detect people
    [bboxes,scores] = detect(peopleDetector,rgb2gray(frame));
    [bboxes,scores] = selectStrongestBbox(bboxes,scores);
    
    goodTrack = find(scores>minThrsld);
    goodBboxes = bboxes(goodTrack,:);
    goodScores = scores(goodTrack);
    
    % Display result
    if goodBboxes > 0
        [rgb,label,pos] = coloredBboxes(frame,goodBboxes,goodScores,nnet,label);
        rgb = insertText(rgb,[pos(1) pos(2)],char(label));
        step(videoPlayer,rgb);
    else
        step(videoPlayer,frame);
    end
    
    waitbar(count/seqLen, w, sprintf('Frame %d on %d',count,seqLen));
    
    drawnow();
    count = count+1;
    cont = ~isDone(videoFReader) && isOpen(videoPlayer); % Break condition
end

release(videoFReader);
release(videoPlayer);
delete(w);
