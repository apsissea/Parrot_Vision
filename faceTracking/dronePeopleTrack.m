function dronePeopleTrack()
%DRONEPEOPLETRACK Summary of this function goes here
%   Detailed explanation goes here
%%
clear global;
%% Params
debug = 1;
rez = 0.5;
minThrsld = 10;
videoFileName = '../Footage/Bebop2_20180313082905+0100.mp4';
%videoFileName = '../Footage/Bebop2_20180310155738+0100.mp4';

%% Objects
videoFReader = vision.VideoFileReader(videoFileName,'VideoOutputDataType','uint8');
videoPlayer = vision.DeployableVideoPlayer('Name','Toto a velo');
peopleDetector = peopleDetectorACF;
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
nnet = alexnet;

%% Init
global rgb;

global isFound;
global points;
global oldPoints;
global new_points;
global ref_points;

global bboxes;
global scores;
global label;
global status;
global trackArea;

status = 'No target';
frameCount = 1;
initTrack = 0;
minTrackNb = 15;
trkwidth = 150;
densityScore = 0;
color = {'r','y','g'};

%% Debug
if debug == 1
    for frameCount = 1:250
        step(videoFReader);
        disp(frameCount);
    end
end

%% Start loop
cont = true;
while cont
    tic
    frame = imresize(step(videoFReader),rez);
    bboxPool = {[0 0 0 0], [0], [0], [{''}]};

    detectPeople();
    track();
    displayResult();
    
    step(videoPlayer,rgb);
    drawnow();
    
    frameCount = frameCount+1;
    cont = isOpen(videoPlayer) && ~isDone(videoFReader); % Break condition
end
disp('Video EOF !');

%% Functions
    function detectPeople()
        % Detect people
        [bboxes,scores] = detect(peopleDetector,rgb2gray(frame));
        [bboxes,scores] = selectStrongestBbox(bboxes,scores); % prevent overlapping bboxes
        goodTrack = find(scores>minThrsld);
        bboxes = bboxes(goodTrack,:);
        scores = scores(goodTrack);
        for i = 1:size(bboxes,1)
            coloredBboxes(bboxes(i,:),scores(i));
        end
    end

    function coloredBboxes(bbox,score)
        densityScore = surclassScoreByTrackers(bbox);
        score = score + (densityScore*100);
        if score<50
            scoreId = 1;
        elseif score>50 && score<75
            scoreId = 2;
        else
            scoreId = 3;
        end
        
        im = imcrop(frame,bbox);
        label = classify(nnet,imresize(im,[227,227])); % Classify
        pool = {bbox, score, scoreId, cellstr(label)};
        bboxPool = [bboxPool; pool];
    
    end

    function lockAndTrack()
        [~,id] = sort(scores,'descend');
        points = detectMinEigenFeatures(rgb2gray(frame), 'ROI', bboxes(id(1),:));
        points = points.Location;
        initialize(pointTracker, points, frame);
    end

    function track()
        if initTrack == 1
            
            [points, isFound] = step(pointTracker, rgb);
            isFound = and(trackZone(trackArea,0),isFound); %kill the points
            new_points = points(isFound,:);
            if isempty(oldPoints)
                oldPoints = points;
            end
            ref_points = oldPoints(isFound,:);
                
            if size(new_points,1) < minTrackNb
                initTrack = 0;
                release(pointTracker)
                points = [];
                oldPoints = [];
                new_points = [];
                oldPoints = [];
                status = 'Target losted';
            else
                xform = estimateGeometricTransform(...
                    ref_points, new_points, 'similarity', 'MaxDistance', 4);
                bboxPoints = bbox2points(trackArea(1, :));
                bboxPoints = transformPointsForward(xform, bboxPoints);
                trackArea = [bboxPoints(1,1),bboxPoints(1,2),trkwidth,trkwidth];
            end
            
            oldPoints = points;
            
        elseif any(cell2mat(bboxPool(:,3))>2)
            lockAndTrack();
            initTrack = 1;
            x = round(mean(points(:,1))-(trkwidth/2));
            y = round(mean(points(:,2))-(trkwidth/2));
            trackArea = [x,y,trkwidth,trkwidth];
            status = strcat('Target locked -- ',cellstr(label));
        end
    end

    function goodPointsId = trackZone(bbox,offset)
        bbox = [bbox(1)-offset,bbox(2)-offset,bbox(3)+offset,bbox(4)+offset];
        goodPointsId = points(:,1) > bbox(1) & points(:,1) < bbox(1)+bbox(3) & ...
            points(:,2) > bbox(2) & points(:,2) < bbox(2)+bbox(4);
    end

    function densityScore = surclassScoreByTrackers(bbox)
        if ~isempty(points)
            goodPointsId = trackZone(bbox,0);
            densityScore = numel(points(goodPointsId,:))/numel(points);
        else
            densityScore = 0;
        end
    end

    function displayResult()
        % Display result
        rgb = frame;
        if size(bboxPool,1)>1
            for id = 2:size(bboxPool,1)
                bbox = bboxPool{id,1};
                rgb = insertText(rgb,[bbox(1) bbox(2)],bboxPool{id,4},'BoxColor',color{bboxPool{id,3}});
                rgb = insertObjectAnnotation(rgb,'rectangle',bbox,bboxPool{id,2},'Color',color{bboxPool{id,3}});
            end
        end
        if ~isempty(new_points)
            rgb = insertMarker(rgb,new_points);
            %rgb = insertMarker(rgb,oldPoints,'Color','white');
            rgb = insertText(rgb,[mean(new_points(:,1)) mean(new_points(:,2))],...
                strcat(num2str(size(new_points,1)),'Pts - area score: ',num2str(densityScore)));
            rgb = insertShape(rgb,'rectangle',trackArea);
        end
        rgb = insertText(rgb,[size(rgb,1),size(rgb,2)/2],status);
        rgb = insertText(rgb,[0 0],['Frame ',num2str(frameCount),' - ',num2str(1/toc),' FPS']);
    end

end

