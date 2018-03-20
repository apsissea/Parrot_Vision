function dronePeopleTrack()
%DRONEPEOPLETRACK Summary of this function goes here
%   Detailed explanation goes here
%%
clear global;
%% Params
debug = 0;
rez = 0.5;
minThrsld = 30;
videoFileName = 'Bebop2_20180313082905+0100.mp4';
%videoFileName = 'Bebop2_20180310155738+0100.mp4';

%% Objects
videoFReader = vision.VideoFileReader(videoFileName,'VideoOutputDataType','uint8');
videoPlayer = vision.DeployableVideoPlayer('Name','Toto a velo');
peopleDetector = peopleDetectorACF;
nnet = alexnet;
%% Init
global rgb;
global points;
global isFound;
global bboxes;
global scores;
global label;
status = 'No target';
frameCount = 1;
initTrack = 0;
minTrackNb = 15;
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
densityScore = 0;

%% Debug
if debug == 1
    for frameCount = 1:800
        step(videoFReader);
        disp(frameCount);
    end
end

%% Start loop
cont = true;
while cont
    tic
    frame = imresize(step(videoFReader),rez);
    
    % Detect people
    [bboxes,scores] = detect(peopleDetector,rgb2gray(frame));
    [bboxes,scores] = selectStrongestBbox(bboxes,scores); % prevent overlapping bboxes
    
    goodTrack = find(scores>minThrsld);
    bboxes = bboxes(goodTrack,:);
    scores = scores(goodTrack);
    
    % Display result
    if goodTrack > 0
        rgb = frame;
        for i = 1:numel(goodTrack)
            rgb = coloredBboxes(rgb,bboxes(i,:),scores(i));
        end
        rgb = insertText(rgb,[0 0],['Frame ',num2str(frameCount),' - ',num2str(1/toc),' FPS']);
    else
        rgb = insertText(frame,[0 0],['Frame ',num2str(frameCount),' - ',num2str(1/toc),' FPS']);
        if ~isempty(points)
            rgb = insertMarker(rgb,points);
        end
    end
    
    track();
    
    rgb = insertText(rgb,[size(rgb,1),size(rgb,2)/2],status);
    step(videoPlayer,rgb);
    drawnow();
    
    frameCount = frameCount+1;
    cont = isOpen(videoPlayer) && ~isDone(videoFReader); % Break condition
end
disp('Video EOF !');


%% Functions
    function rgb = coloredBboxes(rgb,bbox,score)
        color = {'r','y','g'};
        densityScore = surclassScoreByTrackers(bbox);
        score = score + (densityScore*100);
        if score<50
            i = 1;
        elseif score>50 && score<75
            i = 2;
        else
            i = 3;
        end
        
        im = imcrop(rgb,bbox);
        label = classify(nnet,imresize(im,[227,227])); % Classify
        rgb = insertText(rgb,[bbox(1) bbox(2)],char(label),'BoxColor',color{i});
        rgb = insertObjectAnnotation(rgb,'rectangle',bbox,score,'Color',color{i});
        
        if i == 3
            if initTrack == 0
                rgb = lockAndTrack(rgb);
                initTrack = 1;
                status = strcat('Target locked -- ',cellstr(label));
            else
                [~,id] = sort(scores,'descend');
                tmpPoints = detectMinEigenFeatures(rgb2gray(rgb), 'ROI', bboxes(id(1),:));
                points = [points ; tmpPoints.Location]; %%%%%TODO%%%%%%%
                [points, isFound] = step(pointTracker, rgb);
            end
        end
    end

    function rgb = lockAndTrack(rgb)
        [~,id] = sort(scores,'descend');
        points = detectMinEigenFeatures(rgb2gray(rgb), 'ROI', bboxes(id(1),:));
        points = points.Location;
        initialize(pointTracker, points, rgb);
        rgb = insertMarker(rgb,points);
    end

    function track()
        if initTrack == 1
            [points, isFound] = step(pointTracker, rgb);
            points = points(isFound,:);
            if ~isempty(points)
                rgb = insertMarker(rgb,points);
                rgb = insertText(rgb,[mean(points(:,1)) mean(points(:,2))],...
                    strcat(num2str(size(points,1)),'Pts - area score: ',num2str(densityScore)));
            end
            if size(points,1) < minTrackNb
                initTrack = 0;
                release(pointTracker)
                status = 'Target losted';
            end
        end
    end

    function densityScore = surclassScoreByTrackers(bbox)
        if ~isempty(points)
            goodPointsId = points(:,1) > bbox(1) & points(:,1) < bbox(1)+bbox(3) & ...
                points(:,2) > bbox(2) & points(:,2) < bbox(2)+bbox(4);
            densityScore = numel(points(goodPointsId,:))/numel(points);
        else
            densityScore = 0;
        end
    end
end

