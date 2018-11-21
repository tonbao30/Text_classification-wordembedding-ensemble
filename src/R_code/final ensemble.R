folder <- getwd()  
file_list <- list.files(pattern="*.txt")                              

# read in each .txt file in file_list and rbind them into a data frame called data 

compData <- data.frame(A= character(), B= character())
for (eachfile in file_list){
  dat <- read.table(file=eachfile, sep=" ", quote="", comment.char="")
  compData <- rbind(compData,dat)
}

compData <- data.frame(compData)

id <- as.character(unique(compData$V1))

tail(names(sort(table(compData$V1))), 1)

predict_label <- vector()

for (i in seq(1,length(id))) {
  predict_label[i] <- names(sort(summary(as.factor(compData[compData$V1 == id[i],2])), decreasing=T)[1])
}

final <- data.frame(cbind(id,predict_label))
write.table(final, file = "MyData",col.names = F,row.names = F,quote = F)
