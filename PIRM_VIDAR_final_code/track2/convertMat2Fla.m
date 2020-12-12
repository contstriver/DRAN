clc;
clear all;




data_dir = '../track2_final_test_data/';
addpath('./matlab');

save_dir = './final_track2_data2/';
mkdir(save_dir);
data_name = dir(fullfile(data_dir,'*.mat'));


for i=1:size(data_name,1) 
    
    name_data = strcat(data_dir, data_name(i).name)
	a_temp = struct2cell(load(name_data, 'data'));
    data = cell2mat(a_temp);
    load('hyper.mat');
     
    hyper.I = data;
    hyper.HDR = HDR;
    hyper.HDR.samples = 480;
    hyper.HDR.lines = 240;
    save_name = strcat(strcat(save_dir, data_name(i).name(1:end-4)),'.fla');
    FLAwrite(save_name, hyper)
   
end  
