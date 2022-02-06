#clear working space
x <- ls() 
if (length(x)!=0) { 
  rm(list=ls()) 
}
library(Matrix)

# read in profiles
df1 <- read.csv("2018.csv")
df2 <- read.csv("2018_solar.csv")
df <- df1 - df2
as.numeric(df[1,])

# model parameters
act<-5 #number of actions available
periods<-48 #total number of time slots in a day
battery_capacity<-800000*2 #capacity of the battery when fully charged
actions<-c(0,50000,100000,150000,200000) #amount of discharge assoc with avail actions

# matrix of constraint coefficients
MM<-matrix(0,2*periods+1,periods*act+1)
M<- as(MM, "sparseMatrix")
act<-5 # number of possible actions
for(r in 1:periods){
  j<-1
  M[r,1]<-1
  for(c in (act*(r-1)+2):(act*(r-1)+act+1)){
    
    M[r,c]<-actions[j]
    M[periods+r,c]<-1
    M[2*periods+1,c]<-actions[j]
    j<-j+1
  }
}


library(slam)   #package needed by package Rglpk
library(Rglpk)  #package containing the solver

#coefficents for the objective function
obj <- c(1,rep(0,act*periods))

# constraint matrix
mat <- M

# inequality vector
dir <- c(rep(">=",periods),rep("==",periods),"<=")

# binary/continuous output constraints
varType<-c("C", rep("B",act*periods))

# repeat for each day in year
savings <- c()

for (i in 1:365){

  k = i
  if(k %in% c(6,21,56,70,267,294,322,329,336,343,363,364)){
    savings <- c(savings,NA)
    print(i)
 }else{
  rhs <- c(as.numeric(df[i,]),rep(1,periods),battery_capacity)
  ans<-Rglpk_solve_LP(obj, mat, dir, rhs, max=FALSE,types=varType,control = list(lp_time_limit = 0.05 ))
  ansv<-ans$solution-c(ans$optimum,rep(0,act*periods))
  y<-M%*%ansv
  savings <- c(savings,ans$optimum)
  print(i)
  }
}


write.csv(savings, "2018_bin_savings.csv")
