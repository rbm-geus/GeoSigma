function [DataOut,Info] = ReadSurfer7(name)

fr = matlab.io.datastore.DsFileReader(name);
disp('Reading GRD File')

% first 4 bytes for file type
[d,~] = read(fr,4,'OutputType','char');
disp(['FileType - ',d]);

if(strcmp(string(d),"DSRB"))
    disp('File format supported!');
else
    error('File format not supported!');
end

%Read Size of Header Section and Version number
[HeaderSize,~] = read(fr,4,'OutputType','int32');
[HeaderVersion,~] = read(fr,4,'OutputType','int32');

 %Read Section Name (expected: GRID)
 [SectionName,~] = read(fr,4,'OutputType','char');

if SectionName == 'GRID'
disp('Reading Grid Section')
end

 %sec2 should be the integer 72, which is the length of the grid section
 [sec2,~] = read(fr,4,'OutputType','int32');

 %Grid Size
 [sec3,~] = read(fr,8,'OutputType','int32');
 disp(['Grid Size: ',num2str(sec3(1)),' rows and ',num2str(sec3(2)),' columns']);
 GridRows = sec3(1);
 GridColumns = sec3(2);
 
 %Grid Corner Location
 [sec4,~] = read(fr,16,'OutputType','double');
 disp(['xLL: ', num2str(sec4(1)),' ','yLL: ', num2str(sec4(2))]);
 GridxLL = sec4(1);
 GridyLL = sec4(2);
 
 %Grid Resolution
 [sec5,~] = read(fr,16,'OutputType','double');
 disp(['x-size: ', num2str(sec5(1)),'m, ','y-size: ', num2str(sec5(2)),'m']);
 GridResolutionX = sec5(1);
 GridResolutionY = sec5(2);

 %Minimum and Maximum Depths
 [sec6,~] = read(fr,16,'OutputType','double');
 Max_Z = sec6(2);
 Min_Z = sec6(1);

 disp(['Min. Z: ', num2str(sec6(1)),'m, ','Max. Z: ', num2str(sec6(2)),'m']);

 %Rotation of Grid (usually is 0)
 [GridRotation,~] = read(fr,8,'OutputType','double');

 %BlankValue
 [BlankValue,~] = read(fr,8,'OutputType','double');

 %Read Section Name (expected: DATA)
 [SectionName2,~] = read(fr,4,'OutputType','char');

 %Read Section Length in Bytes
 [DataSectionLength,~] = read(fr,4,'OutputType','int32');
 disp(['Data Contains ',num2str(DataSectionLength/1024),' kilobytes']);

 disp(['Reading Data'])
 [Data,~] = read(fr,DataSectionLength,'OutputType','double');
 Data(Data==BlankValue) = NaN;
 Z = reshape(Data,[GridColumns,GridRows])';

 DataOut = Z;

 Info.UTM_X = GridxLL+(0:GridResolutionX:(GridColumns-1)*100);
 Info.UTM_Y = GridyLL+(0:GridResolutionY:(GridRows-1)*100);
 
 Info.BlankValue = BlankValue;

 Info.xLL = GridxLL;
 Info.yLL = GridyLL;

 Info.Cols = GridColumns;
 Info.Rows = GridRows;

 Info.MaxZ = Max_Z;
 Info.MinZ = Min_Z;

 Info.xResolution = GridResolutionX;
 Info.yResolution = GridResolutionY;

 Info.GridRotation = GridRotation;
 disp(['File Loaded Successfully'])
 end