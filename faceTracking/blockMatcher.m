if ~exist('videoFReader','var')
    videoFReader = vision.VideoFileReader('Bebop2_20180313082905+0100.mp4');
end

rez = 0.5;
blockSize = 15;
meta = info(videoFReader);

videoPlayer = vision.VideoPlayer('Position',[1136 57 meta.VideoSize*rez]);
step(videoPlayer,imresize(step(videoFReader),rez));
old_frame = imresize(step(videoFReader),rez);

hbm = vision.BlockMatcher('ReferenceFrameSource','Input port','BlockSize',[blockSize blockSize]);
hbm.OutputValue = 'Magnitude-squared';
halphablend = vision.AlphaBlender;

[X,Y] = meshgrid(1:blockSize:size(old_frame,2),1:blockSize:size(old_frame,1));  

while ~isDone(videoFReader) && isOpen(videoPlayer)
    frame = imresize(step(videoFReader),rez);
    motion = hbm(rgb2gray(old_frame),rgb2gray(frame));
    %step(videoPlayer,frame);
    imshow(frame);
    hold on
    quiver(X(:),Y(:),real(motion(:)),imag(motion(:)),0)
    hold off
    drawnow
end
