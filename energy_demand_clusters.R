# required libraries
library(RColorBrewer)
library(factoextra)

# reading in 2019 data
load <- read.csv("2019.csv")
dow_2019 <- read.csv("2019dow.csv", sep = ",")
dow_2019 <- dow_2019[,5]
weekdays.no = dow_2019
months_2019 <- read.csv("2019months.csv")

#Function to plot profiles by weekday with average
plot.day <- function(day_char, data, color_no){
  day_values <- data[weekdays.no == color_no,]
  day <- day_values[1,]
  day_var <- as.numeric(day)
  plot(day_var, type = 'l', col = "grey",ylim = c(0,1500000), main = day_char, ylab = "Load (kVA)", xaxt = 'n',
       xlab = "Time")
  axis(1, at=1:48, labels=times)
  total = day_var
  
  for (i in 2:nrow(day_values)){
    day <- day_values[i,]
    day_var = as.numeric(day)
    lines(day_var, type = 'l', col = "grey")
    total = total + day_var
  }
  
  colors <- (brewer.pal(7,"Spectral"))
  lines(total/nrow(day_values), type = 'l', col = colors[color_no], lwd = 5)
}

# Plotting profiles by weekday
par(mfrow = c(2,4))
Weekdays = c("Monday", "Tuesday","Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
times <- 1:48

for (i in 1:7){
  plot.day(Weekdays[i],load, i)
}

#Function to plot profiles by month with average

plot.month <- function(month_no, month_char, vis.data){
  day_values <- vis.data[months_2019 == month_no,]
  day <- day_values[1,]
  day_var = as.numeric(day)
  plot(day_var, type = 'l', col = "grey",ylim = c(0,1500000), main = month_char, ylab = "Load (kVA)", xaxt = 'n',
       xlab = "Time")
  axis(1, at=1:48, labels=times)
  total = day_var
  
  for (i in 2:nrow(day_values)){
    day <- day_values[i,]
    day_var = as.numeric(day)
    lines(day_var, type = 'l', col = "grey")
    total = total + day_var
  }
  
  colors <- (brewer.pal(11,"Spectral"))
  colors <- c(colors, colors[1])
  lines(total/nrow(day_values), type = 'l', col = colors[month_no], lwd = 5)
}

# Plotting the profiles by month
par(mfrow = c(2,3))
month_chars = c("January", "February","March", "April", "May", "June", "July", "August", "September", "October", "November", "December")

for (i in 1:12){
  plot.month(i,month_chars[i], load)
}

## plotting demand across a year to view the change in magnitude
par(mfrow = c(1,1))
plot(as.numeric(as.matrix(t(load))), type = 'l')
abline(1000000,0, col = "red", lwd = 5)

# function to plot the weekday averages
plot.day.averages <- function(){
  
  for (i in 1:7){
    day_no = i  
    day_values <- load[weekdays.no == day_no,]
    day <- day_values[1,]
    day_var = as.numeric(day)
    total = day_var
    
    for (i in 2:nrow(day_values)){
      day <- day_values[i,]
      day_var = as.numeric(day)
      total = total + day_var
    }
    
    colors <- (brewer.pal(7,"Spectral"))
    
    if ( day_no == 1){
      plot(total/nrow(day_values), type = 'l', col = colors[day_no], lwd = 5, xlim = c(0,70), ylim = c(300000,1200000))
    }else{
      lines(total/nrow(day_values), type = 'l', col = colors[day_no], lwd = 5)
    }
  }
  legend("topright", Weekdays, col = colors, lwd = 5)
}

# plotting the weekday averages
par(mfrow = c(1,1))
plot.day.averages()

#Applying PCA to the data
max.vals <- apply(load,1,max)
min.vals <- apply(load,1,min)
normalized <- (load - min.vals)/(max.vals - min.vals)
pca.out <- princomp(normalized)

#Plotting the PCA Variance
par(mfrow = c(1,1))
Variance<-(pca.out$sdev)^2
bp <-barplot((Variance/sum(Variance))[1:10], ylim = c(0,1))
labels <- as.character(round(Variance/sum(Variance),4)*100)[1:10]
text(bp,(Variance/sum(Variance))[1:10]+0.04,labels, cex=1, pos=3)

#Plotting first two components coloured by weekday
scores <- (pca.out$scores[,1:2])
colors <- (brewer.pal(7,"Spectral"))
plot(scores, pch = weekdays.no, col = weekdays.no, type = 'n')
text(scores[weekdays.no == 1,],label=1,col=colors[1])
text(scores[weekdays.no == 2,],label=2,col=colors[2])
text(scores[weekdays.no == 3,],label=3,col=colors[3])
text(scores[weekdays.no == 4,],label=4,col=colors[4])
text(scores[weekdays.no == 5,],label=5,col=colors[5])
text(scores[weekdays.no == 6,],label=6,col=colors[6])
text(scores[weekdays.no == 7,],label=7,col=colors[7])

# storing the first 5 components
data.red <- (pca.out$scores[,1:5])


# DBSCAN: Using neighbour distances to find eps for minpts = 10 
dbscan::kNNdistplot(data.red[,1:2], k =  10)
abline(h = 0.15, lty = 2)

# Fitting DBSCAN
x <- data.red[,1:2]
db <- fpc::dbscan(x, eps = 0.2, MinPts =10 )

#Viewing DBSCAN clusters
fviz_cluster(db, data.red[,1:2], geom = "point")

# removing outliers from the data
cluster.no = (db$cluster != 0)
weekdays.no = weekdays.no[cluster.no]
normalized <- normalized[cluster.no,]

# applying pca to the data without outliers
pca.out <- princomp(normalized)

#Plotting the PCA Variance
par(mfrow = c(1,1))
Variance<-(pca.out$sdev)^2
bp <-barplot((Variance/sum(Variance))[1:10], ylim = c(0,1))
labels <- as.character(round(Variance/sum(Variance),4)*100)[1:10]
text(bp,(Variance/sum(Variance))[1:10]+0.04,labels, cex=1, pos=3)

#Plotting first two components
scores <- (pca.out$scores[,1:2])
plot(scores, pch = weekdays.no, col = weekdays.no, type = 'n')
text(scores[weekdays.no == 1,],label=1,col=colors[1])
text(scores[weekdays.no == 2,],label=2,col=colors[2])
text(scores[weekdays.no == 3,],label=3,col=colors[3])
text(scores[weekdays.no == 4,],label=4,col=colors[4])
text(scores[weekdays.no == 5,],label=5,col=colors[5])
text(scores[weekdays.no == 6,],label=6,col=colors[6])
text(scores[weekdays.no == 7,],label=7,col=colors[7])

# storing the first 5 components to be used for clustering 
data.red <- (pca.out$scores[,1:5])

##NBCLUST

#Using muliple methods to find the number of clusters in the data
library(NbClust)
res.nb <- NbClust(data.red, distance = "euclidean",min.nc = 2, max.nc 
                  = 10, method = "complete", index ="all")

par(mfrow = c(1,1))
hist(res.nb$Best.nc[1,],breaks=0:10,col="lightblue",xlab="Optimal number of clusters")


##K-MEANS
library(ClusterR)

#plotting wss
k.max <- 15 # Maximal number of clusters
wss <- sapply(1:k.max, 
              function(k){kmeans(data.red, k, nstart=25 )$tot.withinss})

plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")
abline(v = 5, lty =2)

#plotting silhouette vs k
k.max <- 15
sil <- rep(0, k.max)

for(i in 2:k.max){
  km.res <- kmeans(data.red, centers = i, nstart = 25)
  ss <- silhouette(km.res$cluster, dist(data.red))
  sil[i] <- mean(ss[, 3])
}

plot(1:k.max, sil, type = "b", pch = 19, 
     frame = FALSE, xlab = "Number of clusters k")
abline(v = which.max(sil), lty = 2)

#Plotting the gap statistic
gap_stat <- clusGap(data.red, FUN = kmeans, nstart = 25, K.max = 10, B = 50)
plot(gap_stat, frame = FALSE, xlab = "Number of clusters k") 
abline(v = 5, lty = 2)

#fitting k-means with 5 clusters
km.out <- kmeans(data.red, 5, nstart = 25)

pca.plot <- function(clusters){
  colors <- brewer.pal(6,"Spectral")
  plot(scores, pch = km.out$cluster, col = km.out$cluster, type = 'n')
  plot(scores, pch = weekdays.no[cluster.no], col = weekdays.no[cluster.no], type = 'n')
  text(scores[weekdays.no == 1,],label=1,col=colors[clusters[weekdays.no == 1]])
  text(scores[weekdays.no == 2,],label=2,col=colors[clusters[weekdays.no == 2]])
  text(scores[weekdays.no == 3,],label=3,col=colors[clusters[weekdays.no == 3]])
  text(scores[weekdays.no == 4,],label=4,col=colors[clusters[weekdays.no == 4]])
  text(scores[weekdays.no == 5,],label=5,col=colors[clusters[weekdays.no == 5]])
  text(scores[weekdays.no == 6,],label=6,col=colors[clusters[weekdays.no == 6]])
  text(scores[weekdays.no == 7,],label=7,col=colors[clusters[weekdays.no == 7]])
}

pca.plot(km.out$cluster)




average_corr = function(clusters){
  corrs <- c(NA,NA,NA,NA)
  
  for (i in 1:5){
    
    cluster_no = i  
    
    if (nrow(norm[clusters == i,]) != 1){
    
      mat <- cor(t(norm[clusters == i,]))
      corrs[i] <- mean(mat[upper.tri(mat,diag=FALSE)])
  }
  
  mean(corrs)
}}

average_corr(km.out$cluster)

##K_MEDIODS

#plotting the silhouette width
k.max <- 15
sil <- rep(0, k.max)

for(i in 2:k.max){
  pam.out<- pam(data.red,i)
  sil[i] <- pam.out$silinfo$avg.width
}

plot(1:k.max, sil, type = "b", pch = 19, 
     frame = FALSE, xlab = "Number of clusters k")
abline(v = which.max(sil), lty = 2)

#plotting gap statistic
gap_stat <- clusGap(data.red, FUN = pam, K.max = 10, B = 50)

plot(gap_stat, frame = FALSE, xlab = "Number of clusters k") 
abline(v = 5, lty = 2)

#fitting k-mediods with 2 clusters
pam.out <- pam(data.red, 4)

average_corr(pam.out$cluster)

#pca.plot(pam.out$cluster)

##HIERARCHICAL CLUSTERING
#COMPLETE LINKAGE

#fit and plot dendrogram
dist.out <- dist(data.red, method = "euclidean")
hc <- hclust(dist.out, method = "complete")
plot(hc, labels = F,-1)
rect.hclust(hc, k = 5, border = 2:5) 

#work out conphenetic correlation
cor(dist.out,cophenetic(hc))

#plot silhouette distance vs k
k.max <- 15
sil <- rep(0, k.max)

for(i in 2:k.max){
  sil.sum<- summary(silhouette(cutree(hc,i),dist.out))
  sil[i] <- sil.sum$avg.width
}

plot(1:k.max, sil, type = "b", pch = 19, 
     frame = FALSE, xlab = "Number of clusters k")
abline(v = which.max(sil), lty = 2)

#Viewing the clusters over the year
clusters <- cutree(hc,5)

average_corr(clusters)

#pca.plot(clusters)

#SINGLE LINKAGE

#fit and plot dendrogram
dist.out <- dist(data.red, method = "euclidean")
hc <- hclust(dist.out, method = "single")
plot(hc, labels = F,-1)
rect.hclust(hc, k = 5, border = 2:3) 

#work out conphenetic correlation
cor(dist.out,cophenetic(hc))

#plot silhouette distance vs k
k.max <- 15
sil <- rep(0, k.max)

for(i in 2:k.max){
  sil.sum<- summary(silhouette(cutree(hc,i),dist.out))
  sil[i] <- sil.sum$avg.width
}

plot(1:k.max, sil, type = "b", pch = 19, 
     frame = FALSE, xlab = "Number of clusters k")
abline(v = which.max(sil), lty = 2)

#Viewing the clusters over the year
clusters <- cutree(hc,5)

average_corr(clusters)

pca.plot(clusters)

#AVERAGE LINKAGE

#fit and plot dendrogram
dist.out <- dist(data.red, method = "euclidean")
hc <- hclust(dist.out, method = "average")
plot(hc, labels = F,-1)
rect.hclust(hc, k = 2, border = 2:3) 

#work out conphenetic correlation
cor(dist.out,cophenetic(hc))

#plot silhouette distance vs k
k.max <- 15
sil <- rep(0, k.max)

for(i in 2:k.max){
  sil.sum<- summary(silhouette(cutree(hc,i),dist.out))
  sil[i] <- sil.sum$avg.width
}

plot(1:k.max, sil, type = "b", pch = 19, 
     frame = FALSE, xlab = "Number of clusters k")
abline(v = which.max(sil), lty = 2)

#Viewing the clusters over the year
clusters <- cutree(hc,4)

average_corr(clusters)

pca.plot(clusters)

#CENTROID LINKAGE

#fit and plot dendrogram
dist.out <- dist(data.red, method = "euclidean")
hc <- hclust(dist.out, method = "centroid")
plot(hc, labels = F,-1)
rect.hclust(hc, k = 2, border = 2:3) 

#work out conphenetic correlation
cor(dist.out,cophenetic(hc))

#plot silhouette distance vs k
k.max <- 15
sil <- rep(0, k.max)

for(i in 2:k.max){
  sil.sum<- summary(silhouette(cutree(hc,i),dist.out))
  sil[i] <- sil.sum$avg.width
}

plot(1:k.max, sil, type = "b", pch = 19, 
     frame = FALSE, xlab = "Number of clusters k")
abline(v = which.max(sil), lty = 2)

#Viewing the clusters over the year
clusters <- cutree(hc,4)

average_corr(clusters)

pca.plot(clusters)




#Plotting BIC vs k for GMM clustering
opt_gmm = Optimal_Clusters_GMM(data.red, max_clusters = 10, criterion = "BIC", 
                               
                               dist_mode = "maha_dist", seed_mode = "random_subset",
                               
                               km_iter = 10, em_iter = 10, var_floor = 1e-10, 
                               
                               plot_data = T)

#fitting the GMM model
gmm = GMM(data.red, 4, dist_mode = "maha_dist", seed_mode = "random_subset", km_iter = 10,
          
          em_iter = 10, verbose = F)   

pr = predict_GMM(data.red, gmm$centroids, gmm$covariance_matrices, gmm$weights)

average_corr(pr$cluster_labels)

pca.plot(pr$cluster_labels)


incomplete.days = c(41,42,43,72,74,75,76,77,78,79,80,81,188,189,227,228,289,291,292,338,339,340,341,342,343,344,345,346) 
temp_2019 <- read.csv("Temp_2019.csv")
high_temp = temp_2019[!(1:365 %in% incomplete.days),3]
high_temp = high_temp[-length(high_temp)]

high_temp

#To plot profiles in cluster with average
par(mfrow = c(1,1))
which.cluster = 4

#which.cluster = 5
cluster.no <- km.out$cluster[1:337] == which.cluster
#cluster.no <- km.out$cluster[1:337] %in% c(2,3,4)

time.vals.48 <- c("00:00","00:30","01:00","01:30","02:00","02:30",
                  "03:00","03:30","04:00","04:30","05:00",
                  "05:30","06:00","06:30","07:00","07:30","08:00",
                  "08:30","09:00","09:30","10:00","10:30",
                  "11:00","11:30","12:00","12:30","13:00","13:30",
                  "14:00","14:30","15:00","15:30","16:00",
                  "16:30","17:00","17:30","18:00","18:30","19:00",
                  "19:30","20:00","20:30","21:00","21:30",
                  "22:00","22:30","23:00","23:30")

profiles <- norm[1:337,]
profile.temp <- high_temp

profiles <- profiles[cluster.no,]
profile.temp <- profile.temp[cluster.no]

plot(0,ylim = c(0,1),xlim = c(0,48), xaxt = 'n', type = 'n',xlab = "", ylab = "")

for (j in sort(unique(profile.temp))){
  #for (j in c(14)){  
  
  profiles.sub <- profiles[profile.temp == j,]
  
  #if(nrow(profiles.sub)>0){
  if(!is.null(nrow(profiles.sub))){
    profile = as.numeric(as.character(profiles.sub[1,]))
    #plot(profile, type = 'l', col = alpha("red",0.5),ylim = c(0,1), xaxt = 'n')
    #axis(1, at=1:48, labels=time.vals.48)
    
    total = 0
    
    for (i in 1:nrow(profiles.sub)){
      profile <- as.numeric(as.character(profiles.sub[i,]))
      #lines(profile, type = 'l', col = alpha("red",0.5))
      total = total + profile
    }
    
    colors <- rainbow(23)
    lines(total/nrow(profiles.sub), type = 'l', col = alpha(colors[j-13],0.5), lwd = 2, xaxt = 'n')
    axis(1, at=1:48, labels=time.vals.48)
    
  }
}

legend("topright",legend = sort(unique(profile.temp))[c(2,4,6,8,10,12,14,16,18,20,22)],col =colors[c(2,4,6,8,10,12,14,16,18,20,22)], lwd = 2,cex = 0.7)
grid(nx=48,ny =10)
#colors <- brewer.pal(6,"Spectral")
#lines(total/nrow(profiles), type = 'l', col = colors[which.cluster], lwd = 5)






















