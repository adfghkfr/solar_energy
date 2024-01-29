solar=read.csv("D:/solarenergy/train.csv")
solar
str(solar)
#change the Date into year,month and day to predict
#try to use my kmean-XGBoost prediction to do this competition
#How to deal with the column Module and NA in Temp_m?

#####�B�z�ܼ�Date#####
#2020/6/9
library(lubridate)
solar$Date <- format(as.Date(solar$Date,'%Y/%m/%d'),'%Y-%m-%d')
solar$Date

#���}�x�s��ƪ��~���
year=year(solar$Date)
month=month(solar$Date)
day=mday(solar$Date)
dim(solar)

solar1=data.frame(solar,year,month,day)
solar1
#change integer into numeric
solar1$Generation=as.numeric(as.character(solar1$Generation))
solar1$Irradiance_m=as.numeric(as.character(solar1$Irradiance_m))
solar1$day=as.numeric(as.character(solar1$day))
str(solar1)

solar2<-subset(solar1,select=-c(ID,Date))
solar2


#####Module#####
##�N���O�ܼƧQ��onehot encoding���N##
library(dplyr)
solar %>% count("Module")
##AUO PM060MW3 320W##2106##
##AUO PM060MW3 325W##20##
##MM60-6RT-300##1142##
##SEC-6M-60A-295##316##
pmax=rep(0,3584)#�p�ȿ�X
vmp=rep(0,3584)#�p�ȹq��
lmp=rep(0,3584)#�p�ȹq�y
voc=rep(0,3584)#�}���q��
lsc=rep(0,3584)#�u���q�y
per=rep(0,3584)#�Ҳծį�
for(i in 1:nrow(solar2)){
  if(solar2[i,10]=="AUO PM060MW3 320W"){
    pmax[i]=320
    vmp[i]=33.48
    lmp[i]=9.56
    voc[i]=40.9
    lsc[i]=10.24
    per[i]=19.2
  }
  else if(solar2[i,10]=="AUO PM060MW3 325W"){
    pmax[i]=325
    vmp[i]=33.66
    lmp[i]=9.66
    voc[i]=41.1
    lsc[i]=10.35
    per[i]=19.5
  }
  
  else if(solar2[i,10]=="MM60-6RT-300"){
    pmax[i]=300
    vmp[i]=32.61
    lmp[i]=9.2
    voc[i]=38.97
    lsc[i]=9.68
    per[i]=18.44
  }
  else{
    pmax[i]=295
    vmp[i]=31.6
    lmp[i]=9.34
    voc[i]=39.4
    lsc[i]=9.85
    per[i]=17.74
  }
}
tail(pmax)
solar2=data.frame(solar2,pmax,vmp,lmp,voc,lsc,per)
str(solar2)


library(caret)
dummy=dummyVars("~.",data=solar2)
solar2=data.frame(predict(dummy,newdata=solar2))
solar2
str(solar2)
names(solar2)[[10]]="Module_320W"
names(solar2)[[11]]="Module_325W"
names(solar2)[[12]]="Module_300"
names(solar2)[[13]]="Module_295"
solar2$Module_320W

solar.data<-subset(solar2,select=-Generation)
solar.data

#####�B�z�ܼƯʥ���(missing value)#####
sum(is.na(solar.data$Temp_m))#1458
#��1/3�ƶq��missing value(1458/3584)
#�Q�Φh���t�ɪk
#���w�ʥ�����ƬOMAR #����i���Q��L�������ʥ��Ȯt�ɥi�_�v�T�ǽT�v
library(mice)
library(VIM)
mice.plot=aggr(solar.data,col=c("navyblue","red"),numbers=TRUE)
#we know that Temp_m, Irradiance, Temp has missing value

#�h����???�]Multiple Imputation�^�O�@����_���`��???��???�z�ʥ��Ȫ���k�C
#��???�@???�]�t�ʥ��Ȫ�???�u�����ͦ��@???���㪺???�u���C
#�C??????�u�������ʥ�???�u�λX�S�d����k???��???
#mice()�q�@�ӥ]�t�ʥ��ƾڪ��ƾڮw�}�l�A��^�@�ӥ]�tm�ӧ���ƾڶ���H�A�C�ӧ���ƾڶ����O�q�L
#���l�ƾڮؤ����ʥ��ƾڶi��t�ɦӥͦ���
#with()�̦���C�ӧ���ƾڶ����βέp�ҫ�
#pool()�N�o�ǳ�W�����R���G��X���@�յ��G
str(solar.data)
imputed.data=mice(solar.data,method="cart")
imp.data1=complete(imputed.data,1)
imp.data2=complete(imputed.data,2)
imp.data3=complete(imputed.data,3)
imp.data4=complete(imputed.data,4)
imp.data5=complete(imputed.data,5)
#�Q�Φh���t�ɲ��ͤF5�Ӹ�ƶ��A��ܨ䤤�@�Ӷi�汵�U�Ӫ��w��

imp.data=function(t){
  complete(imputed.data,t)
}

imp.data(1)
plot=aggr(imp.data(1))#�T�w���s�bmissing data

#####kmeans-XGBoost�ҫ��إ�#####
library(plyr)
wss<-vector()
for (i in 2:15){ 
  wss[i] <- sum(kmeans(imp.data(1),i)$withinss)}

par(mfrow=c(1,1))
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="Assessing the Optimal Number of Clusters with the Elbow Method",
     pch=20, cex=2) 
##�̨Τ��s��:k=5##

###train data+validation data����������###
clus.ans<-kmeans(imp.data(1),5)
cluster=clus.ans$cluster
str(imp.data(1))

###�N�E�����G��Jimp.data�A��Ѧ�train+validation
library(e1071)
imp.data1=data.frame(imp.data(1),cluster)
tail(imp.data1)
smp.size=floor(0.8*nrow(imp.data1))
set.seed(1)                     
train.ind=sample(seq_len(nrow(imp.data1)), smp.size)
train=imp.data1[train.ind,] 
vali=imp.data1[-train.ind,]
train
vali
dim(train) #2867 22
dim(vali)  #717 22


###�Ntrain data�Pvalidation data�������w�����ܼ�Generation�A�}�l�إߤ����ҫ�
train.class.data=subset(train)
vali.class.data=subset(vali)
str(vali.class.data)


###�إߤ����ҫ�
library(xgboost)
library(Matrix)
library(data.table)
trainclass<-data.table(train.class.data,keep.rownames=F)
valiclass<-data.table(vali.class.data,keep.rownames=F)
trainclass$cluster<-as.numeric(train.class.data$cluster)-1  ##for cv.model
valiclass$cluster<-as.numeric(vali.class.data$cluster)-1  ##for cv.model
sparse_matrix <- sparse.model.matrix(cluster~.-cluster,data=trainclass) 
str(trainclass)
output_vector = trainclass[,cluster]
sparse_matrix1 <- sparse.model.matrix(cluster~.-cluster,data =valiclass)
output_vector1 = valiclass[,cluster]
dtrain = xgb.DMatrix(data=sparse_matrix,label=output_vector)
dtest = xgb.DMatrix(data=sparse_matrix1,label=output_vector1)
params <- list(booster = "gbtree", objective = "multi:softprob", num_class = 5, 
               eval_metric = "mlogloss")
xgbcv <- xgb.cv(params = params, data =dtrain, nrounds = 50, nfold = 10, showsd = TRUE, 
                stratified = TRUE, 
                print_every_n = 10, early_stop_round = 20, maximize = FALSE, prediction = TRUE)
classification_error <- function(conf_mat) {
  conf_mat = as.matrix(conf_mat)
  error = 1 - sum(diag(conf_mat)) / sum(conf_mat)
  return (error)
}
xgb_train_preds <- data.frame(xgbcv$pred) %>% mutate(max = max.col(., ties.method = "last"),
                                                     label = output_vector + 1)
head(xgb_train_preds)
xgb_conf_mat <- table(true = output_vector + 1, pred = xgb_train_preds$max)
xgb_conf_mat
cat("XGB Training Classification Error Rate:", classification_error(xgb_conf_mat), "\n")
library(caret)
xgb_conf_mat_2 <- confusionMatrix(factor(xgb_train_preds$label),
                                  factor(xgb_train_preds$max),mode = "everything")
print(xgb_conf_mat_2)#0.9995#1#0.9998#1#1#
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 50)

###validation model
xgb_val_preds <- predict(xgb_model, newdata = dtest)

xgb_val_out <- matrix(xgb_val_preds, nrow = 5, ncol = length(xgb_val_preds) / 5) %>% 
  t() %>%
  data.frame() %>%
  mutate(max = max.col(., ties.method = "last"), label = output_vector1 + 1) 

xgb_val_conf <- table(true = output_vector1 + 1, pred = xgb_val_out$max)
xgb_val_conf
cat("XGB Validation Classification Error Rate:", classification_error(xgb_val_conf), "\n")
xgb_val_conf2 <- confusionMatrix(factor(xgb_val_out$label),
                                 factor(xgb_val_out$max),mode = "everything")
print(xgb_val_conf2)#1#1#1#1#1

###�N�����ҫ����������G��Jtrain data�Pvalidation data
trainmodel.clus=xgb_train_preds$max
valimodel.clus=xgb_val_out$max
trainmodel=data.frame(train.class.data,trainmodel.clus)
str(trainmodel) #2867 23
names(trainmodel)[[23]]="clus"
trainmodel
valimodel=data.frame(vali.class.data,valimodel.clus)
names(valimodel)[[23]]="clus"
valimodel

###�]���إߤ����ҫ����ɭԧ�w���ܼ�Generation�R���F�A�]�������s�[�JGeneration
predict=rbind(trainmodel,valimodel)
imp.data1=data.frame(imp.data1,solar2$Generation)
dim(imp.data1)#3584 23
str(imp.data1)
names(imp.data1)[[23]]="Generation"

predict.data=left_join(predict,imp.data1,by=c("Temp_m"="Temp_m","Irradiance"="Irradiance","Capacity"="Capacity",
                                              "Lat"="Lat","Lon"="Lon","Angle"="Angle","Irradiance_m"="Irradiance_m",
                                              "Temp"="Temp","year"="year","month"="month","day"="day",
                                              "Module_320W"="Module_320W",
                                              "Module_325W"="Module_325W","Module_300"="Module_300",
                                              "Module_295"="Module_295","cluster"="cluster","pmax"="pmax",
                                              "vmp"="vmp","lmp"="lmp","voc"="voc","lsc"="lsc",
                                              "per"="per"))
dim(predict.data)#3584 24
str(predict.data)

###�Ntrain data�Pvalidation data���O���s�إ߸��
clus=function(i){
  subset(predict.data,clus==i)
}
list(dim(clus(1)),dim(clus(2)),dim(clus(3)),dim(clus(4)),dim(clus(5)))
#1620 634 259 484 587

###���F�إ߹w���ҫ��R��cluster�ܼƻPtrainmodel.clus�ܼ�
clus1=function(i){
  subset(clus(i),select=-c(cluster,clus))
}
list(dim(clus1(1)),dim(clus1(2)),dim(clus1(3)),dim(clus1(4)),dim(clus1(5)))
#22���ܼ�

clus1(1)$Generation
clus1(2)$Generation
clus1(3)$Generation
clus1(4)$Generation
clus1(5)$Generation

###��cluster1�إ߹w���ҫ�
clus1(1)
smp.size=floor(0.8*nrow(clus1(1))) 
set.seed(1)                     
train.ind=sample(seq_len(nrow(clus1(1))), smp.size)
trainclus1=clus1(1)[train.ind,] 
valiclus1=clus1(1)[-train.ind,]
dim(trainclus1) #1296 22
dim(valiclus1)  #324 22
str(trainclus1)
traingene1<-trainclus1$Generation
valigene1<-valiclus1$Generation

###xgboost model###
train_data1 <- data.matrix(trainclus1[, -22]) 
train_label1 <- trainclus1$Generation
dtrain1 <- xgb.DMatrix(data = train_data1, label= train_label1)

test_data1 <- data.matrix(valiclus1[, -22])
test_label1 <- valiclus1$Generation
dtest1 <- xgb.DMatrix(data = test_data1, label= test_label1)
params1 <- list(booster = "gbtree", objective = "reg:squarederror",
                eval_metric = "rmse",subsample=0.6,gamma=0.2,min_child_weight=20,
                colsample_bytree=1,max_depth=5,eta=0.1)
#eta:learning rate
#Calculate of folds for cross-validation#
xgbcv1 <- xgb.cv(params = params1, data =dtrain1, nrounds = 100, nfold = 10, showsd = TRUE, 
                 stratified = TRUE, 
                 print_every_n = 20, early_stop_round = 20, 
                 maximize = FALSE, prediction = TRUE)

#�إ߼ҫ�#
xgb.model1 = xgb.train(paras=params1,data=dtrain1,
                       nrounds=100)

##���n�S�x�e��##
imp_fearture1 <- xgb.importance(colnames(dtrain1), model = xgb.model1)
print(imp_fearture1)
library(Ckmeans.1d.dp)
xgb.ggplot.importance(imp_fearture1)

##prediction train
train.pred1 = predict(xgb.model1, dtrain1)
postResample(pred = train.pred1, obs = train_label1)
#      RMSE   Rsquared        MAE 
#27.7741106  0.9989121 19.0386299 

##prediction validation
valipred1=predict(xgb.model1,dtest1)
postResample(pred = valipred1, obs = test_label1)
#       RMSE    Rsquared         MAE 
#248.8801893   0.9110908 136.6880412 


###��cluster2�إ߹w���ҫ�
clus1(2)
smp.size=floor(0.8*nrow(clus1(2))) 
set.seed(1)                     
train.ind=sample(seq_len(nrow(clus1(2))), smp.size)
trainclus2=clus1(2)[train.ind,] 
valiclus2=clus1(2)[-train.ind,]
dim(trainclus2) #507
dim(valiclus2)  #127
str(trainclus2)
traingene2<-trainclus2$Generation
valigene2<-valiclus2$Generation

train_data2 <- data.matrix(trainclus2[, -22]) 
train_label2 <- trainclus2$Generation
dtrain2 <- xgb.DMatrix(data = train_data2, label= train_label2)

test_data2 <- data.matrix(valiclus2[, -22])
test_label2 <- valiclus2$Generation
dtest2 <- xgb.DMatrix(data = test_data2, label= test_label2)
params2 <- list(booster = "gbtree", objective = "reg:squarederror",
                eval_metric = "rmse",subsample=0.5,gamma=0.3,min_child_weight=15,
                colsample_bytree=1,max_depth=6,eta=0.01)
#Calculate of folds for cross-validation#
xgbcv2 <- xgb.cv(params = params2, data =dtrain2, nrounds = 100, nfold = 10, showsd = TRUE, 
                 stratified = TRUE, 
                 print_every_n = 20, early_stop_round = 20, 
                 maximize = FALSE, prediction = TRUE)

#�إ߼ҫ�#
xgb.model2 = xgb.train(paras=params2,data=dtrain2,
                       nrounds=100)

##���n�S�x�e��##
imp_fearture2 <- xgb.importance(colnames(dtrain2), model = xgb.model2)
print(imp_fearture2)
xgb.ggplot.importance(imp_fearture2)

##prediction train
train.pred2 = predict(xgb.model2, dtrain2)
postResample(pred = train.pred2, obs = train_label2)
#RMSE:0.092 #R-SQUARED:0.999 #MAE:0.065

##prediction validation
valipred2=predict(xgb.model2,dtest2)
postResample(pred = valipred2, obs = test_label2)
#RMSE:31.749 #R-SQUARED:0.89 #MAE:21.93

###��cluster3�إ߹w���ҫ�
clus1(3)
smp.size=floor(0.8*nrow(clus1(3))) 
set.seed(1)                     
train.ind=sample(seq_len(nrow(clus1(3))), smp.size)
trainclus3=clus1(3)[train.ind,] 
valiclus3=clus1(3)[-train.ind,]
dim(trainclus3) #268
dim(valiclus3)  #68
str(trainclus3)
traingene3<-trainclus3$Generation
valigene3<-valiclus3$Generation

train_data3 <- data.matrix(trainclus3[, -22]) 
train_label3 <- trainclus3$Generation
dtrain3 <- xgb.DMatrix(data = train_data3, label= train_label3)

test_data3 <- data.matrix(valiclus3[, -22])
test_label3 <- valiclus3$Generation
dtest3 <- xgb.DMatrix(data = test_data3, label= test_label3)
params3 <- list(booster = "gbtree", objective = "reg:squarederror",
                eval_metric = "rmse",subsample=0.5,gamma=0.3,min_child_weight=20,
                colsample_bytree=1,max_depth=10,eta=0.01)
#Calculate of folds for cross-validation#
xgbcv3 <- xgb.cv(params = params3, data =dtrain3, nrounds = 100, nfold = 10, showsd = TRUE, 
                 stratified = TRUE, 
                 print_every_n = 20, early_stop_round = 20, 
                 maximize = FALSE, prediction = TRUE)

#�إ߼ҫ�#
xgb.model3 = xgb.train(paras=params3,data=dtrain3,
                       nrounds=100)

##���n�S�x�e��##
imp_fearture3 <- xgb.importance(colnames(dtrain3), model = xgb.model3)
print(imp_fearture3)
library(Ckmeans.1d.dp)
xgb.ggplot.importance(imp_fearture3)

##prediction train
train.pred3 = predict(xgb.model3, dtrain3)
postResample(pred = train.pred3, obs = train_label3)
#RMSE:0.461 #R-SQUARED:0.999 #MAE:0.31

##prediction validation
valipred3=predict(xgb.model3,dtest3)
postResample(pred = valipred3, obs = test_label3)
#RMSE:80.99 #R-SQUARED:0.97 #MAE:60.610

###��cluster4�إ߹w���ҫ�
clus1(4)
smp.size=floor(0.8*nrow(clus1(4))) 
set.seed(1)                     
train.ind=sample(seq_len(nrow(clus1(4))), smp.size)
trainclus4=clus1(4)[train.ind,] 
valiclus4=clus1(4)[-train.ind,]
dim(trainclus4) #913
dim(valiclus4)  #229
str(trainclus4)
traingene4<-trainclus4$Generation
valigene4<-valiclus4$Generation

train_data4 <- data.matrix(trainclus4[, -22]) 
train_label4 <- trainclus4$Generation
dtrain4 <- xgb.DMatrix(data = train_data4, label= train_label4)

test_data4 <- data.matrix(valiclus4[, -22])
test_label4 <- valiclus4$Generation
dtest4 <- xgb.DMatrix(data = test_data4, label= test_label4)
params4 <- list(booster = "gbtree", objective = "reg:squarederror",
                eval_metric = "rmse",subsample=0.5,gamma=0.1,min_child_weight=20,
                colsample_bytree=1,max_depth=10,eta=0.4)
#Calculate of folds for cross-validation#
xgbcv4<- xgb.cv(params = params4, data =dtrain3, nrounds = 100, nfold = 10, showsd = TRUE, 
                 stratified = TRUE, 
                 print_every_n = 20, early_stop_round = 20, 
                 maximize = FALSE, prediction = TRUE)

#�إ߼ҫ�#
xgb.model4 = xgb.train(paras=params4,data=dtrain4,
                       nrounds=100)

##���n�S�x�e��##
imp_fearture4 <- xgb.importance(colnames(dtrain4), model = xgb.model4)
print(imp_fearture4)
library(Ckmeans.1d.dp)
xgb.ggplot.importance(imp_fearture4)

##prediction train
train.pred4 = predict(xgb.model4, dtrain4)
postResample(pred = train.pred4, obs = train_label4)
#RMSE:17.085 #R-SQUARED:0.999 #MAE:11.88

##prediction validation
valipred4=predict(xgb.model4,dtest4)
postResample(pred = valipred4, obs = test_label4)
#RMSE:208.953 #R-SQUARED:0.938 #MAE:149.18


###��cluster5�إ߹w���ҫ�
clus1(5)
smp.size=floor(0.8*nrow(clus1(5))) 
set.seed(1)                     
train.ind=sample(seq_len(nrow(clus1(5))), smp.size)
trainclus5=clus1(5)[train.ind,] 
valiclus5=clus1(5)[-train.ind,]
dim(trainclus5) #913
dim(valiclus5)  #229
str(trainclus5)
traingene5<-trainclus5$Generation
valigene5<-valiclus5$Generation

train_data5 <- data.matrix(trainclus5[, -22]) 
train_label5 <- trainclus5$Generation
dtrain5 <- xgb.DMatrix(data = train_data5, label= train_label5)

test_data5 <- data.matrix(valiclus5[, -22])
test_label5 <- valiclus5$Generation
dtest5 <- xgb.DMatrix(data = test_data5, label= test_label5)
params5 <- list(booster = "gbtree", objective = "reg:squarederror",
                eval_metric = "rmse",subsample=0.5,gamma=0.01,min_child_weight=20,
                colsample_bytree=1,max_depth=10,eta=0.5)
#Calculate of folds for cross-validation#
xgbcv5 <- xgb.cv(params = params5, data =dtrain5, nrounds =200, nfold = 10, showsd = TRUE, 
                 stratified = TRUE, 
                 print_every_n = 20, early_stop_round = 20, 
                 maximize = FALSE, prediction = TRUE)

#�إ߼ҫ�#
xgb.model5 = xgb.train(paras=params5,data=dtrain5,
                       nrounds=100)

##���n�S�x�e��##
imp_fearture5 <- xgb.importance(colnames(dtrain5), model = xgb.model3)
print(imp_fearture5)
library(Ckmeans.1d.dp)
xgb.ggplot.importance(imp_fearture5)

##prediction train
train.pred5 = predict(xgb.model5, dtrain5)
postResample(pred = train.pred5, obs = train_label5)
#RMSE:0.93 #R-SQUARED:0.999 #MAE:0.68

##prediction validation
valipred5=predict(xgb.model5,dtest5)
postResample(pred = valipred5, obs = test_label5)
#RMSE:85.23 #R-SQUARED:0.95 #MAE:53.33

library(MLmetrics)
RMSPE(train.pred5,train_label5)#0.0001
MAPE(train.pred5,train_label5)#0.0001
RMSE(train.pred5,train_label5)#0.079

RMSPE(train.pred4,train_label4)#0.0029
MAPE(train.pred4,train_label4)#0.001
RMSE(train.pred4,train_label4)#2.27

RMSPE(train.pred3,train_label3)#0.009
MAPE(train.pred3,train_label3)#0.0019
RMSE(train.pred3,train_label3)#2.488

RMSPE(train.pred2,train_label2)#0.0466
MAPE(train.pred2,train_label2)#0.021
RMSE(train.pred2,train_label2)#28.54

RMSPE(train.pred1,train_label1)#0.0017
MAPE(train.pred1,train_label1)#0.00098
RMSE(train.pred1,train_label1)#1.026

RMSPE(valipred5,test_label5)#0.099
MAPE(valipred5,test_label5)#0.046
RMSE(valipred5,test_label5)#46.65

RMSPE(valipred4,test_label4)#0.11
MAPE(valipred4,test_label4)#0.061
RMSE(valipred4,test_label4)#115.79

RMSPE(valipred3,test_label3)#0.21
MAPE(valipred3,test_label3)#0.125
RMSE(valipred3,test_label3)#517.7707

RMSPE(valipred2,test_label2)#0.24
MAPE(valipred2,test_label2)#0.1347
RMSE(valipred2,test_label2)#258.7

RMSPE(valipred1,test_label1)#0.11
MAPE(valipred1,test_label1)#0.06
RMSE(valipred1,test_label1)#93.021


####�Q��test data �i����s�ùw��
test.data=read.csv("D:/solarenergy/test.csv")
test.data

#####�B�z�ܼ�Date#####
test.data$Date <- format(as.Date(test.data$Date,'%Y/%m/%d'),'%Y-%m-%d')
test.data$Date
#���}�x�s��ƪ��~���
year=year(test.data$Date)
month=month(test.data$Date)
day=mday(test.data$Date)
dim(test.data) #1539 12

test.data2=data.frame(test.data,year,month,day)
#change integer into numeric
test.data2$Generation=as.numeric(as.character(test.data2$Generation))
test.data2$Irradiance_m=as.numeric(as.character(test.data2$Irradiance_m))
test.data2$day=as.numeric(as.character(test.data2$day))
test.data2$year=as.numeric(as.character(test.data2$year))
test.data2$month=as.numeric(as.character(test.data2$month))
str(test.data2)

test2<-subset(test.data2,select=-c(ID,Date,Generation))#�R�����w�����ܼ�generation
test2
str(test2)

#####Module#####
##�N���O�ܼƧQ��one hot encoding���N##
library(dplyr)
test2 %>% count("Module")
##AUO PM060MW3 320W##988##
##AUO PM060MW3 325W##111##
##MM60-6RT-300##328##
##SEC-6M-60A-295##112##
pmax=rep(0,1539)#�p�ȿ�X
vmp=rep(0,1539)#�p�ȹq��
lmp=rep(0,1539)#�p�ȹq�y
voc=rep(0,1539)#�}���q��
lsc=rep(0,1539)#�u���q�y
per=rep(0,1539)#�Ҳծį�
for(i in 1:nrow(test2)){
  if(test2[i,10]=="AUO PM060MW3 320W"){
    pmax[i]=320
    vmp[i]=33.48
    lmp[i]=9.56
    voc[i]=40.9
    lsc[i]=10.24
    per[i]=19.2
  }
  else if(test2[i,10]=="AUO PM060MW3 325W"){
    pmax[i]=325
    vmp[i]=33.66
    lmp[i]=9.66
    voc[i]=41.1
    lsc[i]=10.35
    per[i]=19.5
  }
  
  else if(test2[i,10]=="MM60-6RT-300"){
    pmax[i]=300
    vmp[i]=32.61
    lmp[i]=9.2
    voc[i]=38.97
    lsc[i]=9.68
    per[i]=18.44
  }
  else{
    pmax[i]=295
    vmp[i]=31.6
    lmp[i]=9.34
    voc[i]=39.4
    lsc[i]=9.85
    per[i]=17.74
  }
}
tail(pmax)
test2=data.frame(test2,pmax,vmp,lmp,voc,lsc,per)
str(test2)
dim(test2)#1539 18


library(caret)
dummy=dummyVars("~.",data=test2)
test3=data.frame(predict(dummy,newdata=test2))
test3
names(test3)[[9]]="Module_320W"
names(test3)[[10]]="Module_325W"
names(test3)[[11]]="Module_300"
names(test3)[[12]]="Module_295"
test3
library(dplyr)
library(BBmisc)
library(clustMixType)
library(psych)
library(moments)

#####�B�z�ܼƯʥ���(missing value)#####
sum(is.na(test3$Temp_m))#440
library(mice)
library(VIM)
mice.plot=aggr(test3,col=c("navyblue","red"),numbers=TRUE)
#we know that Temp_m, Irradiance, Temp has missing value

imputed=mice(test3,method="cart")
impdata1=complete(imputed,1)
impdata2=complete(imputed,2)
impdata3=complete(imputed,3)
impdata4=complete(imputed,4)
impdata5=complete(imputed,5)
#�Q�Φh���t�ɲ��ͤF5�Ӹ�ƶ��A��ܨ䤤�@�Ӷi�汵�U�Ӫ��w��

impdata=function(t){
  complete(imputed,t)
}

impdata(1)
plot=aggr(impdata(1))#�T�w���s�bmissing data


###��J�����ҫ�###
sparse_matrix <- sparse.model.matrix(cluster~.,data=trainclass)[,-1] 
#-1����]�Osparse.model.matrix�|���ͲĤ@��column intercept�A�R���~�ध��ϥ�predict()�A�_�h
#�|���variable���P��error
output_vector = trainclass[,cluster]
sparse_matrix1 <- sparse.model.matrix(cluster~.,data =valiclass)[,-1]
output_vector1 = valiclass[,cluster]
dtrain = xgb.DMatrix(data=sparse_matrix,label=output_vector)
dtest = xgb.DMatrix(data=sparse_matrix1,label=output_vector1)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100)
xgb_val_preds <- predict(xgb_model, newdata = dtest)


testone<-as.data.table(impdata(1))
testone2=as.matrix(testone)
testone3=xgb.DMatrix(testone2)
colnames(testone3)
colnames(dtrain)

testclass<-predict(xgb_model,testone3)
testclass

xgb_val_test <- matrix(testclass, nrow = 5, ncol = length(testclass) / 5) %>% 
  t() %>%
  data.frame() %>%
  mutate(max = max.col(., ties.method = "last"))

classpredict<-xgb_val_test$max
classpredict

class.test=data.frame(impdata(1),classpredict)
class.test

##�N�������G�����h�Ӹ�ƶ��ǳƶi��w��
predictclus1=subset(class.test,classpredict==1)
predictclus2=subset(class.test,classpredict==2)
predictclus3=subset(class.test,classpredict==3)
predictclus4=subset(class.test,classpredict==4)
predictclus5=subset(class.test,classpredict==5)
list(dim(predictclus1),dim(predictclus2),dim(predictclus3),dim(predictclus4),dim(predictclus5))
#164#542#480#272#81

str(predictclus1)
##���Ӹs�i��U�۪�generation�w��
predictclus11<-subset(predictclus1,select=-c(classpredict))
predictclus111<-as.matrix(predictclus11)
str(predictclus111)
sapply(predictclus111, mode)
result1<-predict(xgb.model1,predictclus111)
result1
set1<-data.frame(result1,predictclus11)
names(set1)[[1]]="Generation"
str(set1)

predictclus22<-subset(predictclus2,select=-c(classpredict))
predictclus222<-as.matrix(predictclus22)
str(predictclus222)
sapply(predictclus222, mode)
result2<-predict(xgb.model2,predictclus222)
result2
set2<-data.frame(result2,predictclus22)
names(set2)[[1]]="Generation"

predictclus33<-subset(predictclus3,select=-c(classpredict))
predictclus333<-as.matrix(predictclus33)
str(predictclus333)
sapply(predictclus333, mode)
result3<-predict(xgb.model3,predictclus333)
result3
set3<-data.frame(result3,predictclus33)
names(set3)[[1]]="Generation"

predictclus44<-subset(predictclus4,select=-c(classpredict))
predictclus444<-as.matrix(predictclus44)
str(predictclus444)
sapply(predictclus444, mode)
result4<-predict(xgb.model4,predictclus444)
result4
set4<-data.frame(result4,predictclus44)
names(set4)[[1]]="Generation"

predictclus55<-subset(predictclus5,select=-c(classpredict))
predictclus555<-as.matrix(predictclus55)
str(predictclus555)
sapply(predictclus555, mode)
result5<-predict(xgb.model5,predictclus555)
result5
set5<-data.frame(result5,predictclus55)
names(set5)[[1]]="Generation"

str(set1)
str(set2)
set=rbind(set1,set2,set3,set4,set5)


##�N�w�����G��^test data
dim(test.data)
dim(impdata(1))
ID=test.data$ID
submission=data.frame(impdata(1),ID)
submission
str(submission)
str(set)
str(class.test)
dim(submission)
dim(set)

submiss=left_join(submission,set,by=c("Temp_m"="Temp_m","Irradiance"="Irradiance",
                                      "Capacity"="Capacity","Lat"="Lat","Lon"="Lon",
                                      "Angle"="Angle","Irradiance_m"="Irradiance_m",
                                      "Temp"="Temp"))
submiss

final=subset(submiss,select=c(ID,Generation))
dim(final)
write.csv(final,"D:/solarenergy/sub2.csv")

sub=read.csv("D:/solarenergy/submission.csv")
dim(sub)