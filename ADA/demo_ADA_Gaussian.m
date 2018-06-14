% function experiment_toy_LocalLDA2()

linewidth = 5;
fontsize = 16;
markersize = 12;

% generate two classes from two Gaussian function
nSmp = 200;
nFea  =  2;
delta1 = 10;
delta2 = 0.25;
mu1 = [-2.5 5];
mu2 = [0 0];
sigma1 = diag([delta1 delta2]);
sigma2 = diag([delta1 delta2]);

% Rotation Matrix
% R = [sqrt(2)/2 sqrt(2)/2; -sqrt(2)/2 sqrt(2)/2];
theta = pi/4;
R = [cos(theta) sin(theta); -sin(theta) cos(theta)];

gnd_inst =[];
inst_data = [];
inst_data = [mvnrnd(mu1,sigma1,nSmp); mvnrnd(mu2,sigma2,nSmp)]*R;
gnd_inst =  [ones(nSmp,1);-ones(nSmp,1)];

width= 800;
height = 640; 
figure1 = figure;
axes1 = axes('Parent',figure1,'LineWidth',2,...
    'GridLineStyle','--',...
    'FontWeight','bold',...
    'FontSize',fontsize,...
    'FontName','Times New Roman');
box(axes1,'on');
hold(axes1,'all');
% Create plot
plot(inst_data(find(gnd_inst==1),1),inst_data(find(gnd_inst==1),2),...
    'MarkerSize',markersize,'Marker','o','LineWidth',linewidth,'LineStyle','none',...
    'Color',[1 0 0]);
% Create plot
plot(inst_data(find(gnd_inst==-1),1),inst_data(find(gnd_inst==-1),2),...
    'MarkerSize',markersize,'Marker','o','LineWidth',linewidth,'LineStyle','none',...
    'Color',[0 0 0]);
legend('Positive Instances', 'Negative Instances','Location','NorthWest');
% Create xlabel
xlabel('x_1','FontWeight','bold','FontSize',fontsize,'FontName','Times New Roman',...
    'FontAngle','italic');
% Create ylabel
ylabel('x_2','FontWeight','bold','FontSize',fontsize,'FontName','Times New Roman',...
    'FontAngle','italic');

fea = inst_data;
gnd = gnd_inst;
Ratio = 0.8;
[train_index, test_index] = splitdata_random(gnd, Ratio);

ttrain_data = fea(train_index,:);
ttrain_label = gnd(train_index);
ttest_data = fea(test_index,:);
ttest_label = gnd(test_index);

Num_Kfolds = 5;
cv_index= crossvalind('Kfold',ttrain_label,Num_Kfolds);
t_set = [0.0000001 0.000001 0.00001 0.0001 0.001 0.01 0.1 1];
for parat = 1:length(t_set)
    for fold_id=1:Num_Kfolds
        % The remaning folds as the training dataset
        test_index_cv = find(cv_index ==fold_id);
        train_index_cv = find(cv_index ~=fold_id);
        
        ttrain_data_cv = ttrain_data(train_index_cv,:);
        ttrain_label_cv = ttrain_label(train_index_cv);
        ttest_data_cv = ttrain_data(test_index_cv,:);
        ttest_label_cv = ttrain_label(test_index_cv);
        
        options = [];
        options.IterMax = 20;
        options.t =t_set(parat);
        d =max(gnd);
        % [W_llda, S, obj] = NLocalLDA(ttrain_data_cv, ttrain_label_cv, d, options);
        expa = exp(1);
        [W_llda, S, obj] = NLocalLDA_log(ttrain_data_cv, ttrain_label_cv, d, options,expa);
        
        % LDA
        options_lda = [];
        options_lda.Fisherface = 1;
        [W_lda, eigvalue] = LDA(ttrain_label_cv,options_lda,ttrain_data_cv);
        
        
        figure2 = figure(2);
        axes2 = axes('Parent',figure2,'LineWidth',2,...
            'GridLineStyle','--',...
            'FontWeight','bold',...
            'FontSize',fontsize,...
            'FontName','Times New Roman');
        box(axes2,'on');
        hold(axes2,'all');
        % Create plot
        plot(inst_data(find(gnd_inst==1),1),inst_data(find(gnd_inst==1),2),...
            'MarkerSize',markersize,'Marker','o','LineWidth',linewidth,'LineStyle','none',...
            'Color',[1 0 0]);
        % Create plot
        plot(inst_data(find(gnd_inst==-1),1),inst_data(find(gnd_inst==-1),2),...
            'MarkerSize',markersize,'Marker','o','LineWidth',linewidth,'LineStyle','none',...
            'Color',[0 0 0]);
        
        data_mu = mean(inst_data);
        c_lda = W_lda(2,1)*data_mu(1,1)-W_lda(1,1)*data_mu(1,2);
        c_llda = W_llda(2,1)*data_mu(1,1)-W_llda(1,1)*data_mu(1,2);
        xmin = min(inst_data(:,1));
        xmax = max(inst_data(:,1));
        ymin = min(inst_data(:,2));
        ymax = max(inst_data(:,2));
        
        toy_x = [xmin:0.01: xmax];
        primal_y_lda = (W_lda(2,1)*toy_x+c_lda)./W_lda(1,1);
        primal_y_llda = (W_llda(1,1)*toy_x+c_llda)./W_llda(2,1);
        
        plot(toy_x,primal_y_llda,'-b','LineWidth',linewidth);
        plot(toy_x,primal_y_lda,'--g','LineWidth',linewidth);
        legend('Positive', 'Negative','ADA','LDA','Location','NorthWest');
        set(gcf, 'Position', [200 200 width height]);
        set(gcf, 'PaperPositionMode', 'auto')
        
        fprintf('One experiment is finished.\n');
    end
end

