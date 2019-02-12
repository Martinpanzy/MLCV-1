clear all; close all; 
% Initialisation
init; clc;
%% 1. Data loading/generation
[data_train, data_test] = getData('Caltech');
%[data_train, data_test] = getData_rfcb('Caltech');
%% 2 Random forest
param.num = 50;         % Number of trees
param.depth = 6;        % Depth of each tree
param.splitNum = 10;     % Number of trials in split function
param.split = 'IG';     % Currently support 'information gain' only

% Train Random Forest
trees = growTrees(data_train,param);

% Test Random Forest
testTrees_script;

%% 3. Random forest parameters
%% 3.1 Number of trees
%N = [1,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500];
%N = [1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300]
N = [20,30]
acc_tr = zeros(1,length(N));
acc_te = zeros(1,length(N));
time_tr = zeros(1,length(N));
time_te = zeros(1,length(N));
for i = 1:length(N) % Number of trees, try {1,3,5,10, or 20}
    param.num = N(i);
    param.depth = 5;    % trees depth
    param.splitNum = 10; % Number of trials in split function
    param.split = 'IG'; % Currently support 'information gain' only
  
    % Train Random Forest
    time1 = clock;
    trees = growTrees(data_train, param);
    time2 = clock;
    time_tr(i) = etime(time2, time1);
    testTrees_script_trainset;
    acc_tr(i) = accuracy_rf*100;
    
    % Test Random Forest
    time1 = clock;
    testTrees_script;
    time2 = clock;
    time_te(i) = etime(time2, time1);
    acc_te(i) = accuracy_rf*100;
end

yyaxis left
title('Performance against Number of Trees')
xlabel('Number of Trees')
ylabel('Recognition Accuracy (%)')
set(gca,'FontSize',13)

yyaxis right
ylabel('Time Efficiency (s)')

hold on
yyaxis left
plot(N,acc_tr,'-o',N,acc_te,'--o','LineWidth',2);

yyaxis right
plot(N,time_tr,'-o',N,time_te,'--o','LineWidth',2);

hold off
legend('Training Accuracy','Testing','Training Time','Testing Time');