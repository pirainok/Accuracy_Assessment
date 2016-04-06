import numpy as np
import math

x = np.array([[31,19,0],[0,49,1],[1,9,38]])
area = [288092,4019545,149742]
Wi = [(area[0]/sum(list(area))),(area[1]/sum(list(area))),(area[2]/sum(list(area)))]
Wi_sqr = [(Wi[0]**2),(Wi[1]**2),(Wi[2]**2)]

def sampleAcc(x):
	'''x = numpy matrix of sample data from classification'''
	a = x[0, :].sum(), x[1, :].sum(),x[2,:].sum() #row sums
	xToti_1 = a[0] #to forest row sum map change (i)
	xToti_2 = a[1] #no change row sum map change (i)
	xToti_3 = a[2] #from forest row sum map change (i)
	b = x[:,0].sum(),x[:,1].sum(),x[:,2].sum() #column sums
	xTotj_1 = b[0] #to forest column sum ref. data
	xTotj_2 = b[1] #no change column sum ref. data
	xTotj_3 = b[2] #from forest column sum ref. data
	xTot_ij = x.sum() #total sample points
	userAcc_1 = (x[0,0])/(xToti_1) #user acc. to forest
	userAcc_2 = (x[1,1])/(xToti_2) #user acc. no change
	userAcc_3 = (x[2,2])/(xToti_3) #user acc. from forest 
	prodAcc_1 = (x[0,0])/(xTotj_1) #prod. acc. to forest
	prodAcc_2 = (x[1,1])/(xTotj_2) #prod. acc. no change
	prodAcc_3 = (x[2,2])/(xTotj_3) #prod. acc. from forest
	overAcc = (x.trace(0))/(xTot_ij) #overall acc.
	userAcc = [userAcc_1,userAcc_2,userAcc_3] #lists users acc.[to forest,no change, from forest]
	prodAcc = [prodAcc_1,prodAcc_2,prodAcc_3] #lists producers acc.[to forest, no change, from forest]
	print(userAcc,prodAcc,'overall accuracy = ',overAcc)
	row_1 = x[0,0],x[0,1],x[0,2],a[0] #1st row in new sample confusion matrix
	row_2 = x[1,0],x[1,1],x[1,2],a[1] #2nd row in new sample confusion matrix
	row_3 = x[2,0],x[2,1],x[2,2],a[2] #3rd row in new sample confusion matrix
	row_4 = b[0],b[1],b[2],x.sum() #4th row in new sample confusion matrix
	newArray = np.array([row_1,row_2,row_3,row_4]) #new confusion matrix of sample data
	#print(newArray)
	return newArray
	
def propArray(sampleArray,areaList):
	'''sampleArray = completed confusion matrix of sample data
	areaList = list of areas of each class'''
	Wi = [(areaList[0]/sum(list(areaList))),(areaList[1]/sum(list(areaList))),(areaList[2]/sum(list(areaList)))] #list of prop. area for each class [to forest, no change, from forest]
	totArea = sum(list(areaList)) #total area of classes
	Wi_sqr = [(Wi[0]**2),(Wi[1]**2),(Wi[2]**2)]	
	sampleArray = x
	x00 = (x[0,0]/x[0].sum())*Wi[0] #r0c0 element
	x01 = (x[0,1]/x[0].sum())*Wi[0] #r0c1 element
	x02 = (x[0,2]/x[0].sum())*Wi[0] #r0c2 element
	x03 = x00+x01+x02 #r0c3 element
	x10 = (x[1,0]/x[1].sum())*Wi[1] #r1c0 element
	x11 = (x[1,1]/x[1].sum())*Wi[1] #r1c1 element
	x12 = (x[1,2]/x[1].sum())*Wi[1] #r1c2 element
	x13 = x10+x11+x12 #r1c3 element
	x20 = (x[2,0]/x[2].sum())*Wi[2] #r2c0 element
	x21 = (x[2,1]/x[2].sum())*Wi[2] #r2c1 element
	x22 = (x[2,2]/x[2].sum())*Wi[2] #r2c2 element
	x23 = x20+x21+x22 #r2c3 element
	x30 = x00+x10+x20 #r3c0 element
	x31 = x01+x11+x21 #r3c1 element
	x32 = x02+x12+x22 #r3c2 element
	x33 = x03+x13+x23 #r3c3 element
	propArea = np.array([[x00,x01,x02,x03],[x10,x11,x12,x13],[x20,x21,x22,x23],[x30,x31,x32,x33]]) #confusion matrix for proportional area
	#print(propArea)
	userAcc_0 = propArea[0,0]/propArea[0,3] #users acc. row 0
	userAcc_1 = propArea[1,1]/propArea[1,3] #users acc. row 1
	userAcc_2 = propArea[2,2]/propArea[2,3] #users acc. row 2
	prodAcc_0 = propArea[0,0]/propArea[3,0] #prod.acc. col. 0
	prodAcc_1 = propArea[1,1]/propArea[3,1] #prod.acc. col. 1
	prodAcc_2 = propArea[2,2]/propArea[3,2] #prod.acc. col. 2
	overAcc = propArea[0,0]+propArea[1,1]+propArea[2,2] #overall Acc.
	#print(overAcc)
	#print(userAcc_0,userAcc_1,userAcc_2)
	#print(prodAcc_0,prodAcc_1,prodAcc_2)
	varOverAcc = ((Wi_sqr[0]*userAcc_0)*(1-userAcc_0)/(x[0].sum()-1))+((Wi_sqr[1]*userAcc_1)*(1-userAcc_1)/(x[1].sum()-1))+((Wi_sqr[0]*userAcc_2)*(1-userAcc_2)/(x[2].sum()-1)) #estimated variance of the overall accuracy proportion of area
	#print(varOverAcc)
	varUserAcc_0 = (userAcc_0*(1-userAcc_0)/(x[0].sum())) #estimated variance user acc. row 0
	varUserAcc_1 = (userAcc_1*(1-userAcc_1)/(x[1].sum())) #estimated variance user acc. row 1
	varUserAcc_2 = (userAcc_2*(1-userAcc_2)/(x[2].sum())) #estimated variance user acc. row 2
	#print(varUserAcc_0,varUserAcc_1,varUserAcc_2)
	p_hat_0 = x30
	p_hat_1 = x31
	p_hat_2 = x32
	estAreaToForest = totArea * p_hat_0 #estimated area to forest
	estAreaNoChng = totArea * p_hat_1 #estimated area no change
	estAreaFromFor = totArea * p_hat_2 #estimated area from forest
	#print(estAreaToForest,estAreaNoChng,estAreaFromFor)
	stdErrorProp_0 = math.sqrt((Wi[0]*(Wi[0]*(x[0,0]/x[0].sum()))-(Wi[0]*(x[0,0]/x[0].sum()))**2)/(x[0].sum()-1) + (Wi[0]*(Wi[0]*(x[0,1]/x[0].sum()))-(Wi[0]*(x[0,1]/x[0].sum()))**2)/(x[0].sum()-1) + (Wi[0]*(Wi[0]*(x[0,2]/x[0].sum()))-(Wi[0]*(x[0,2]/x[0].sum()))**2)/(x[0].sum()-1)) #std error of prop. area to forest
	stdErrorProp_1 = math.sqrt((Wi[1]*(Wi[1]*(x[1,0]/x[1].sum()))-(Wi[1]*(x[1,0]/x[1].sum()))**2)/(x[1].sum()-1) + (Wi[1]*(Wi[1]*(x[1,1]/x[1].sum()))-(Wi[1]*(x[1,1]/x[1].sum()))**2)/(x[1].sum()-1) + (Wi[1]*(Wi[1]*(x[1,2]/x[1].sum()))-(Wi[1]*(x[1,2]/x[1].sum()))**2)/(x[1].sum()-1)) #std error of prop. no change
	stdErrorProp_2 = math.sqrt((Wi[2]*(Wi[2]*(x[2,0]/x[2].sum()))-(Wi[2]*(x[2,0]/x[2].sum()))**2)/(x[2].sum()-1) + (Wi[2]*(Wi[2]*(x[2,1]/x[2].sum()))-(Wi[2]*(x[2,1]/x[2].sum()))**2)/(x[2].sum()-1) + (Wi[2]*(Wi[2]*(x[2,2]/x[2].sum()))-(Wi[2]*(x[2,2]/x[2].sum()))**2)/(x[2].sum()-1)) #std error of prop. area from forest
	#print(stdErrorProp_0,stdErrorProp_1,stdErrorProp_2)
	stdErrorEstAreaToFor = totArea * stdErrorProp_0 #std. error of estimated area of change to forest
	stdErrorEstAreaNo = totArea * stdErrorProp_1 #std. error of estimated area of no change
	stdErrorEstAreaFrmFor = totArea * stdErrorProp_2 #std. error of estimated area of change from forest
	#print(stdErrorEstAreaToFor,stdErrorEstAreaNo,stdErrorEstAreaFrmFor)
	Con95Area_toForest = (1.96*math.sqrt(stdErrorEstAreaToFor)) #95% confidence interval for to forest area
	Con95Area_NoChng = (1.96*math.sqrt(stdErrorEstAreaNo)) #95% confidence interval for no change area
	Con95Area_FromFor = (1.96*math.sqrt(stdErrorEstAreaFrmFor))  #95% confidence interval for from forest area
	Con95Over = (1.96*math.sqrt(varOverAcc)) #95% confidence interval for overall accuracy of propotion of area.
	print(Con95Area_toForest,Con95Area_NoChng,Con95Area_toForest)
	
	