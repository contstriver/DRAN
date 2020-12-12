%% Compute the pseudocolor image of an image on the PIRM2018 datasets using
%  the colour matching functions on public_meta_data.mat
%
%% Syntax:
%     Irgb = compute_pseudocolor_image(I);
%     Irgb = compute_pseudocolor_image(I,WhiteBalance);
% 
%% Description
%   Run this script to compute a pseudolour image using the colour matching
%   functions on the propietary_meta_data.mat file.
%
%% Input:
%     I: Data structure containing header information and image cube data 
%         as supplied on an ENVI standard flat file.
%     WhiteBalance: Variable used to determine whether the output image should be
%         white balanced using the white patch algorithm. By default, its set to
%         true.
% 
%% Output:
%     Irgb: Output pseudocolour image
%
%% Examples:
%
%     Read a file from the validation set, compute the pseudocolour
%     image without white balancing and show the resulting image:
%       I =  FLAread('../training/image_20_hr.fla');
%       Irgb = compute_pseudocolor_image(I,false);
%       imshow(Irgb);                     
%
%     Compute the white balanced pseudocolour image and show it in a new
%     figure:
%       Irgb = compute_pseudocolor_image(I);
%       figure;
%       imshow(Irgb);                     
%
% Author: Antonio Robles-Kelly. 
% Version: 1.0.0
% Last Update Date: 6 May 2018

function Irgb = compute_pseudocolor_image(I,WhiteBalance);

%Load the colour matching functions and setup the white balance  variable
load('CMFs2deg.mat');
if ~exist('WhiteBalance','var') ||  WhiteBalance ~= 0
    WhiteBalance = 1;
end

%Compute the pseudocolour image
Irgb = reshape(double(I.I),I.HDR.lines*I.HDR.samples,I.HDR.bands);
Irgb = Irgb*CMFs2deg(:,2:4);
Irgb = Irgb./repmat(max(Irgb,[],1),I.HDR.lines*I.HDR.samples,1);
Irgb = reshape(Irgb, I.HDR.lines, I.HDR.samples, 3);

%White balance if required
if WhiteBalance
    illuminant = illumwhite(Irgb);
    Irgb = Irgb./repmat(reshape(illuminant,1,1,3),I.HDR.lines,I.HDR.samples,1);
    Irgb = Irgb/max(Irgb(:));
end
