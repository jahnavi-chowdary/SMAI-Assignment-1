
Y = importdata('y_data');
% classes = {'Iris-setosa','Iris-versicolor','Iris-virginica'};
classes = [0 1];
%------ Get Number of Vectors in X belonging to each class ----- %

noofclass = length(classes);
vectperclass = zeros(noofclass,1);  %Initializing vectperclass which contains the number of vectors in X belonging to each class.

for (i = 1 : noofclass)
    temp = classes(i);
    Y_ones = ismember(Y,temp);
    vectperclass(i,1) = nnz(Y_ones);
end

%---------- Defining Number of Folds -------- %

 error = zeros(4,5);
 variance = zeros(4,5);
 stddev = zeros(4,5);

for k = 2:5

    foldsmatrix = zeros(noofclass, k);

    for i = 1 : noofclass
        
        noofvects = vectperclass(i, 1);
        
        for j = 1 : k
        
            % Divide the remaining number of vectors by the remaining number of folds.
            foldsize = ceil(noofvects / (k - j + 1));
        
            % Store the fold size.
            foldsmatrix(i,j) = foldsize;
        
            % Update the number of remaining vectors for this category.
            noofvects = noofvects - foldsize;
        
        end
        
    end

    % ------- Apply KNN classification ------- %

    for kn = 1:5
        
        testruns = 10;
        %err = 0;
        err = 0;
        vari = 0;

        for trun = 1:testruns
    
            h = size(X,1);
            randorder = randperm(h);
            randX = X(randorder, :);
            randY = Y(randorder, :);

            X_sorted = [];
            Y_sorted = [];

            % Re-group the vectors according to category.
            for (i = 1 : noofclass)
  
                % Get the next category value.
                tmp = classes(i);
                var = ismember(randY,tmp);
                tmpX = [];
                tmpY = [];
                l=0;
                for j = 1:h
                    if (var(j)==1)
                        l = l+1;
                        tmpX = [tmpX ; randX(j,:)];
                        tmpY = [tmpY ; randY(j,:)];
                    end
                end
                X_sorted = [X_sorted; tmpX];
                Y_sorted = [Y_sorted; tmpY];
            
            end

            %-------- Get the training and testing data ------- %
  
            err_matrix = [];

            for nooffolds = 1:k
                
                X_val = [];
                Y_val = [];
                X_train = [];
                Y_train = [];
                roundno = nooffolds;
                start = 1;

                for i = 1 : noofclass

                    % Get the list of fold sizes for this category as a column vector.
                    foldmat_row = foldsmatrix(i, :);
    
                    % Set the starting index of the first fold for this category.
                    foldstart = start;
    
                    for j = 1 : k
        
                        % Compute the index of the last vector in this fold.
                        foldend = foldstart + foldmat_row(j) - 1;
        
                        % Select all of the vectors in this fold.
                        fold_X = X_sorted(foldstart : foldend, :);
                        fold_Y = Y_sorted(foldstart : foldend, :);
        
                        % If this fold is to be used for validation in this round...
                        if (j == roundno)
                            % Append the vectors to the validation set.
                            X_val = [X_val; fold_X];
                            Y_val = [Y_val; fold_Y];
                            % Otherwise, use the fold for training.
                        else
                            % Append the vectors to the training set.
                            X_train = [X_train; fold_X];
                            Y_train = [Y_train; fold_Y];
                        end
        
                        % Update the starting index of the next fold.
                        foldstart = foldend + 1;
                    
                    end
    
                    % Set the starting index of the next category.
                    start = start + vectperclass(i);   
                
                end

                % ------ KNN CLASSIFIER --------- %
    
                distanc = [];
                labxpl = [];

                final_label = [];
        
                for s = 1:size(X_val,1)
            
                    for t = 1:size(X_train,1)
  
                        dis = 0;
                
                        for p = 1:size(X_train,2)
                    
                            dis = dis + (X_train(t,p) - X_val(s,p)).^2;
                    
                        end
                
                        distanc(s,t) = sqrt(dis);
            
                    end
            
                    [sort_dist , I ] = sort(distanc,2);
                    test = I(s,1:kn);
                    labxpl = [];
            
                    for q = 1:size(test,2)      
                
                        index = test(1,q);
                        labxpl = [labxpl;Y_train(index,1)];
            
                    end
            
                    flag = zeros(1,size(classes,2)) ;
            
                    for m = 1:size(classes,2)
                
                        flag(1,m) = nnz(ismember(labxpl,classes(1,m)));
            
                    end
            
                    [tempo indx] = sort(flag,'descend');
                    final_label = [final_label;classes(1,indx(1,1))]; 
        
                end

                for ab = 1:size(Y_val,1)
            
                    if ((final_label(ab,1) == Y_val(ab,1)))
                        continue;
                    else
                        err = err+1;
                    end
            
                end
                errtemp = err./size(X,1);
                err_matrix = [err_matrix,errtemp];
    
            end
    
        end

        tempora1 =  errtemp/testruns;
        error(k,kn) = tempora1;

        for i = 1 : size(err_matrix,1) 
            vari = vari + (tempora1 - err_matrix(i)).^2; 
        end 

        tempora2 = vari./(testruns - 1);  
        variance(k,kn) = tempora2; 

        tempora3 = sqrt(vari); 
        stddev(k,kn) = tempora3; 

    end
    
end

% accuracy
% stddev

error = error(2:5,:);
stddev = stddev(2:5,:);

xaxis1 = [2 3 4 5];
xaxis2 = [1 2 3 4 5];

% -------- Plots using errorbar function --------- % 

figure, errorbar(xaxis2,error(1,:),stddev(1,:));  %For 2-fold
figure, errorbar(xaxis2,error(2,:),stddev(2,:));  %For 3-fold
figure, errorbar(xaxis2,error(3,:),stddev(3,:));  %For 4-fold
figure, errorbar(xaxis2,error(4,:),stddev(4,:));  %For 5-fold

% ------- X axis -> k-fold , Y axis -> Error , For KNN (1NN - 5NN) ----- %

figure, plot(xaxis1,error(:,1),'b--o',xaxis1,error(:,2),'r--o',xaxis1,error(:,3),'g--o',xaxis1,error(:,4),'c--o',xaxis1,error(:,5),'k--o');
legend('1-NN','2-NN','3-NN','4-NN','5-NN','Location','northwest');

% ------- X axis -> k-fold , Y axis -> Standard deviation , For KNN (1NN - 5NN) ----- %

figure, plot(xaxis1,stddev(:,1),'b--o',xaxis1,stddev(:,2),'r--o',xaxis1,stddev(:,3),'g--o',xaxis1,stddev(:,4),'c--o',xaxis1,stddev(:,5),'k--o');
legend('1-NN','2-NN','3-NN','4-NN','5-NN','Location','northwest');
