% folder with the tif files and the masks are required for this plotting
% teh folder structure
clear all

folder_path='E:\Final_Data_Set_INTERPOLATION_PAPER\';
% sav_path='F:\Frame_interpolation_tests\INT_Amy_fluorescent\16x_interpolated_images\Segmented\';
fold_names=dir(folder_path);
fold_names = fold_names([fold_names(:).isdir]);
fold_names = fold_names(~ismember({fold_names(:).name},{'.','..','SSIMs','16_Ananlysis','Tracks','AP_Results','INTERP','INTERP2','INTERP3','Results_conka_1','Results_conka_2','not_tracked','not_tracked','Input','to_segment','Segmented','REST','INTERP','INTERP2','INTERP3'}));

fold_names = { fold_names.name};

sae=1; % 1 means saving

colrs=['r'; 'b'; 'k'; 'g'; 'c';'m';'y';'b'; 'r'; 'k'; 'g'; 'c';'m';'y']; % for plotting


for io =1%1:size(fold_names,2)-1

% folder path to original images without interpolation
path_h=['E:\Final_Data_Set_INTERPOLATION_PAPER\' fold_names{io} '\interpolated_images\'  ]; 

exp_ground = 'all_images_renamed_final'; % name of the folder containing th ground truth images (which weere used as inputs to the model)

exps={ 'CDFI' 'RIFE' 'FILM' 'LDMVFI'}; % models to check, add LMDVFI once is ready

cha='_cp_masks.tif';%

AP_ALL_Data=cell(3,size(exps,2));

for ikk = 1:size(exps,2)

  AP_ALL_Data{1,ikk} =exps{ikk};

end

for ik1=1:size(exps,2)

filenames=dir(fullfile([path_h exp_ground], ['*' cha]));
f_names = { filenames.name };

filenames2=dir(fullfile([path_h exps{ik1}], ['*' cha]));
f_names2 = { filenames2.name };

AP_mask_Data=cell(size(f_names,2)-1,4);
AP_cell_Data=cell(size(f_names,2)-1,4);

for itt = 1:size(f_names,2)-2%[im_no1:im_no] % [1:343 345:790]% itt=1

    % name1 = replace(f_names{itt+1},cha,'.tif');
    % name2 = replace(f_names2{itt},cha,'.tif');

% I = imread([[path_h exp_ground] '\'  name1]); % figure;imagesc(I)
M = imread([[path_h exp_ground] '\'  f_names{itt+1}]); % figure;imagesc(M)

% IT = imread([[path_h exps{ik1}] '\'  name2]); % figure;imagesc(IT)
MT = imread([[path_h exps{ik1}] '\'  f_names{itt}]); % figure;imagesc(MT)

% check if the 2 images have the same number of indexes

cels1=unique(M(M~=0))';

tru_posi=zeros(1,numel(cels1));
false_neg=zeros(1,numel(cels1));
false_posi=zeros(1,1);

%% false negatives and tru positives
for ik=1:numel(cels1)

I3=double(M==cels1(ik)); % figure;imagesc(I3)

I4=I3.*double(MT); %imagesc(I4)

ind1=mode(I4(I4~=0));

I5=double(I4==ind1);

I5=imdilate(I5,ones(2));

I6=I5-I3; %imagesc(I6)

if sum(sum(I6(I6~=0))) <=  0.4*(sum(sum(I3(I3~=0)))) % true positive

tru_posi(1,ik)=1;

else  % sum(sum(I6(I6~=0))) >= 0.4*(sum(sum(I3(I3~=0))))


false_neg(1,ik)=1;

end
end

%% false positives (doesnt need a loop)
I6=M~=0; %  figure;imagesc(I6)
I6=bwmorph(I6,"fatten",3);
I6=double(~I6);

I7=double(MT).*I6;% figure;imagesc(I7)

if max(I7(:))>0 % there are false positives
    false_posi=numel(unique(I7(I7~=0)));
end

AP_mask=sum(tru_posi)./(sum(tru_posi)+false_posi+sum(false_neg));

if itt==1 
AP_mask_Data{itt,1}='AP_mask';
AP_mask_Data{itt,2}='tru_posi';
AP_mask_Data{itt,3}='false_posi';
AP_mask_Data{itt,4}='false_neg';
end

AP_mask_Data{itt+1,1}=AP_mask;
AP_mask_Data{itt+1,2}=tru_posi;
AP_mask_Data{itt+1,3}=false_posi;
AP_mask_Data{itt+1,4}=false_neg;


%% AP for each mask

cell_false_posi=zeros(1,numel(cels1));% false positives
cell_false_nega=zeros(1,numel(cels1));% false negative
cell_tru_posi=zeros(1,numel(cels1));% 
cell_AP=zeros(1,numel(cels1));

AP_cor_single_cell=nan(2,2);



for ik2=1:numel(cels1)

% identifies the current cell as a binary mask 
IA1=double(M==cels1(ik2)); % % figure;imagesc(IA1)

% if ik2==1 % just to artificially make the second maks with different indexes
% IB1=MT;
% IB1(MT~=0)=IB1(MT~=0)+100;   % figure;imagesc(IB1)
% else
% 
% end

% to get the index under of the current cell in the interpolated image 
% figure;imagesc(IA3)
IA3=IA1.*double(MT);
ind2=mode(IA3(IA3~=0));

if ~isnan(ind2) % the mask exist in the interpolated image
IA4=double(IA3==ind2); % figure;imagesc(IA4)
else % its a false negative
IA4=IA3;
end

%just to artificially make the cell mask with false positives
if ik2==1
IA4=imdilate(IA4,ones(27));
end

IA5=double(IA4)-double(IA1);% substrat to obtain false positives (+1) and negatives (-1) 
% figure;imagesc(IA5)

IA6=IA4+IA1; % figure;imagesc(IA6)% counts the overlap, true positives, as 2's

if sum(IA5(:))==0
    %min(IA5(IA5~=0))~=-1 && max(IA5(IA5~=0))~=1

cell_AP(1,ik2)=1;

else %max(IA5(:))~=0

cell_false_posi(1,ik2)=sum(sum(IA5==1));% false positives
cell_false_nega(1,ik2)=sum(sum(IA5==-1));% false negative
cell_tru_posi(1,ik2)=sum(sum(IA6==2));% 

cell_AP(1,ik2)=sum(sum(IA6==2))./(sum(sum(IA6==2))+sum(sum(IA5==-1))+sum(sum(IA5==1)));


end
end

% correction for false positives
I6=M~=0; % imagesc(I6)
I6=bwmorph(I6,"fatten",3);
I6=double(~I6);

I7=double(MT).*I6;% imagesc(I7)

if max(I7(:))>0 % there are false positives

    false_posi_correction=numel(unique(I7(I7~=0)));

end

cell_AP(cell_AP==0)=nan;
% 
%  XS=sum(XS1,'omitnan')/(length(XS1)+AP_cell_Data{2,6});
% % AP_cor_single_cell(1,1)=mean(XS,'omitnan');
% % AP_cor_single_cell(1,2)=std(AP_cell_Data{2,2},'omitnan');

%% rough correction for false positives (does not take into consideration the size of the false positives)
AP_cor_single_cell(1,1)=sum(cell_AP,'omitnan')./(sum(~isnan(cell_AP))+false_posi_correction); % Average corrected AP of time point
AP_cor_single_cell(1,2)=std(cell_AP,'omitnan');
%% AP without false positives correction
AP_cor_single_cell(2,1)=sum(cell_AP,'omitnan')/(sum(~isnan(cell_AP)));
AP_cor_single_cell(2,2)=std(cell_AP,'omitnan');



if itt==1

AP_cell_Data{itt,1}='AP_cor_single_cell';
AP_cell_Data{itt,2}='cell_AP';
AP_cell_Data{itt,3}='cell_tru_posi';
AP_cell_Data{itt,4}='cell_false_nega';
AP_cell_Data{itt,5}='cell_false_posi';
AP_cell_Data{itt,6}='false_posi_correction';

end

AP_cell_Data{itt+1,1}=AP_cor_single_cell;
AP_cell_Data{itt+1,2}=cell_AP;
AP_cell_Data{itt+1,3}=cell_tru_posi;
AP_cell_Data{itt+1,4}=cell_false_nega;
AP_cell_Data{itt+1,5}=cell_false_posi;
AP_cell_Data{itt+1,6}=false_posi_correction;

end


 AP_ALL_Data{2,ik1}=AP_cell_Data;
 AP_ALL_Data{3,ik1}=AP_mask_Data;


save(['AP_Data_' fold_names{io} '_' exps{ik1} ],"AP_cell_Data","AP_mask_Data")

 AP1=zeros(2,size(AP_cell_Data,1)-1);
for iu=2:size(AP_cell_Data,1)

    AP1(1,iu-1) = AP_cell_Data{iu,1}(1,1);
    AP1(2,iu-1) = AP_cell_Data{iu,1}(1,2);
    
end

% plot data 
titel1=[ fold_names{io} '_AP_plot_' exps{ik1}];
% Sample data
f1=figure(1);
x =  1:size(AP1(1,:),2);
y = AP1(1,:); % Mean values
std_dev = AP1(2,:); % Standard deviations

% Plotting the data with error bars
errorbar(x, y, std_dev,'r');
xlabel('Time points');
ylabel('AP');
title([exps{ik1} ' AP per whole image and single cell mask']);
ylim([0.4 1])
hold on

plot(cell2mat(AP_mask_Data(2:end,1))','b')

hold off

   if sae==1
saveas(f1,titel1,'pdf')
   else
   end

% close all
end


save(['AP_ALL_Data_' fold_names{io}  '_' exps{ik1}],"AP_ALL_Data")

  APmean = nan(1,size(AP_ALL_Data{2,2},1)-1);
  APstd = nan(1,size(AP_ALL_Data{2,2},1)-1);

for ih=1:size(AP_ALL_Data,2)

res1=AP_ALL_Data{2,ih};

for iu=2:size(res1,1)

    APmean(ih,iu-1) = res1{iu,1}(1,1);
    APstd(ih,iu-1) = res1{iu,1}(1,2);
    
end
end



%% plot all models together with STD
% plot data 
titel2=[fold_names{io} '_AP_STD_ALL'];
f2=figure(2);
colrs=['r'; 'b'; 'k'; 'g'; 'c';'m';'y';'b'; 'r'; 'k'; 'g'; 'c';'m';'y']; % for plotting
for ij=1:4
x =  1:size(APmean,2);
y = APmean(ij,:); % Mean values
std_dev = APstd(ij,:); % Standard deviations

% Plotting the data with error bars
errorbar(x, y, std_dev,colrs(ij));
xlabel('Time points');
ylabel('AP');
title('AP per single cell mask');
hold on
end
hold off
 xlim([1 size(APmean,2)])  
 ylim([0.4 1]) 
 title(titel2,Interpreter="none");
 legend({'CDFI','RIFE','FLIM','LDMVFI'}, 'Location', 'northeast', 'FontSize', 12);


   if sae==1
saveas(f2,titel2,'pdf')
   else
   end
% close all

%% plot with 95% confidence intervals, uses all masks in eahc time point to produce the biologcal replicates!
titel3=[fold_names{io} '_AP_CI'];% io=1

ploto1=cell(1,3);

for ih=1:size(AP_ALL_Data,2)

res2=AP_ALL_Data{2,ih};

ploto=nan(150,size(AP_ALL_Data{2,2},1)-1);

for ik=2:size(AP_ALL_Data{2,2},1)
ploto(1:size(res2{ik,2}',1),ik-1)=res2{ik,2}'; % imagesc(ploto)
end

ploto1{1,ih}=ploto;

end

meanOfChambers = nan(4, size(ploto1{1,2},2));
for i=1:4 % loop to obtain the mean of each lane

AA=ploto1{1,i};
AA(AA==0)=nan;

    meanOfChambers(i,:) = mean(AA,"omitmissing");
end
t=1:length(meanOfChambers(i,:));

f3=figure(3);

for ik4=1:4  % loop to plot the average curve for each cluster
    sem = std(ploto1{1,ik4}, [], 1, 'omitnan')./sqrt(size(ploto1{1,ik4},1));
    hold on
    s1 = shadedErrorBarV2(t, meanOfChambers(ik4,:), 2*sem, 'lineprops', colrs(ik4));
    hold on  
    %pause
end
hold off

      xlim([1 size(t,2)]) 
      ylim([0.5 1])
 title(titel3,Interpreter="none");
 legend({'','','','CDFI','','','','RIFE','','','', 'FLIM','','','','LDMVFI'}, 'Location', 'northeast', 'FontSize', 12);

   if sae==1
saveas(f3,titel3,'pdf')
   else
   end

 close all
end

