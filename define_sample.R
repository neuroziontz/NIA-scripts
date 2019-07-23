library(dplyr)
library(readxl)
library(xlsx)

# Ensure Windows-Mac compatibility
if (.Platform$OS.type=="unix") {
  rootname <- paste(.Platform$file.sep, "Volumes/NIAIRP2", sep="")
} else { # if Windows
  rootname <- paste("Q:", .Platform$file.sep, sep="")
}

cogpath <- file.path(rootname,'LBN','BABS','cogstore','cog1')
datapath <- file.path(cogpath,'home','jacob','entorhinal-connectivity-and-tau','data')

readtaudata <- function(fname) {
  da <- as.data.frame(read_excel(fname, skip=1))
  colnames(da) <- make.names(colnames(da), unique=T)
  da$blsaid <- as.numeric(substr(da$image.path,21,24))
  da$blsavi <- as.numeric(substr(da$image.path,26,27)) + as.numeric(substr(da$image.path,29,29))/10
  da
}

# PET spreadsheet
petda_fname <- file.path(datapath,'PETstatus_09Nov2018.xlsx')
petda <- as.data.frame(read_excel(petda_fname))
petda <- petda[c("blsaid","blsavi","AV1451date","PIBdate","sex","race","educ_years","apoe4","av1451currdx","dx","AV1451age")]

# SUVR spreadsheet
tauda_fname <- file.path(datapath,'ROI_pvc_SUVR_Nov092018.xlsx')
tauda <- readtaudata(tauda_fname)
tauda <- tauda[c("blsaid","blsavi","entorhinal","Braak.I.II","Braak.III.IV","Braak.V.VI","inferior.temporal.gyrus","amygdala","hippocampus","parahippocampal")]
tauda <- merge(tauda, petda, by=c("blsaid","blsavi"), all.x=T)
tauda <- tauda[order(tauda$blsaid,tauda$blsavi),]
tauda$tau_date <- as.Date(tauda[["AV1451date"]], "%m/%d/%y") # convert to date
tauda$pib_date <- as.Date(tauda[["PIBdate"]], "%d-%b-%y") # convert to date
tauda$taupib_diff <- difftime(tauda$tau_date, tauda$pib_date, units="days") # difference in days between tau and pib scans
tauda_base <- as.data.frame(tauda %>% group_by(blsaid) %>% slice(which.min(blsavi))) # get baseline visits only
tauda_base <- subset(tauda_base, tauda_base$av1451currdx==0 | tauda_base$av1451currdx==0.5) # only CN and MCI participants

# resting state spreadsheet
rs_fname <- file.path(datapath,'rsdata_from_archive_assessed_08-Apr-2019_usable_normalized_data_09-Apr-2019.xlsx')
rsda <- as.data.frame(read_excel(rs_fname))
rsda <- rsda[!is.na(rsda$id),]
rsda <- rsda[rsda$usable==1,]
rsda$rs_date <- as.Date(rsda[["Date_Collected"]], "%Y%m%d") # convert to date
colnames(rsda)[colnames(rsda)=="id"] <- "blsaid"
colnames(rsda)[colnames(rsda)=="vis"] <- "rs_blsavi"
rsda <- rsda[c("blsaid","rs_blsavi","rs_date","long_sess_num")]

# get resting state scans within 2 years of tau scans
#colnames(tauda_base)[colnames(tauda_base)=="blsavi"] <- "tau_blsavi"
taursda <- merge(tauda_base, rsda, by=c("blsaid"))
taursda$taurs_diff <- difftime(taursda$tau_date, taursda$rs_date, units="days") # difference in days between tau and rs scans
#taursda <- taursda[order(taursda$blsaid, taursda$rs_blsavi),]
# spreadsheet with all rs scans for participants with av1451 scans, includes some without corresponding scans
#write.xlsx(taursda, file.path(datapath,'taurs_long_sample.xlsx'), row.names=F)

# pair baseline AV1451 scan with closest rs scan
taursda <- as.data.frame(taursda %>% group_by(blsaid) %>% slice(which.min(abs(taurs_diff)))) # rs scan closest to tau scan
taursda <- taursda[taursda$taurs_diff<=730,] # no participants with rs scans more than 2 years prior to tau scan
taursda <- taursda[taursda$taurs_diff>=-730,] # no participants with rs scans more than 2 years after tau scan
colnames(taursda)[colnames(taursda)=="rs_date"] <- "rs_date_base"
taursda <- taursda[c("blsaid","blsavi","rs_blsavi","tau_date","pib_date","rs_date_base","taupib_diff","taurs_diff","AV1451age","sex","race","educ_years","apoe4","av1451currdx","dx",
                     "entorhinal","Braak.I.II","Braak.III.IV","Braak.V.VI","inferior.temporal.gyrus","amygdala","hippocampus","parahippocampal")]

# PiB spreadsheet
pibda_fname <- file.path(datapath,'csv_output_PiB_DVR_concise_20190207.csv')
pibda <- as.data.frame(read.csv(pibda_fname))
pibda$pibgroup <- as.numeric(pibda$pibgroup.f=='PiB+')
#colnames(pibda)[colnames(pibda)=="blsavi"] <- "pib_blsavi"
pibda <- pibda[c("blsaid","blsavi","pibgroup")]
taurspibda <- merge(taursda, pibda, by=c("blsaid","blsavi"), all.x=T)
# impute pibgroup from scans close in time to tau scan - TO DO: implement this prior to rs data using criteria for automatic impution
taurspibda$pibgroup[taurspibda$blsaid=="4706"] <- 0
taurspibda$pibgroup[taurspibda$blsaid=="4931"] <- 0

# add mean fd for each participant
taurspibda$idvi <- paste0(sprintf("%04d",taurspibda$blsaid),"_",sprintf("%02d",taurspibda$rs_blsavi),"-0")
fd_data <- list()
for (idvi in taurspibda$idvi) {
  fd_fname <- file.path(entconpath,'data','rsfmri_images',paste0("BLSA_",idvi,"_10"),'fd_0.5mm.txt')
  fd_vector <- as.list(read.delim(fd_fname, header=F))
  fd_data[[idvi]] <- mean(fd_vector$V1)
}
taurspibda$fd <- unlist(fd_data)

# center independent variables for model interpretability
independent_vars <- c("AV1451age","sex","pibgroup","apoe4","entorhinal","taurs_diff","fd","Braak.I.II","Braak.III.IV","Braak.V.VI","inferior.temporal.gyrus","amygdala","hippocampus","parahippocampal")
for (var in independent_vars) {
  taurspibda[,paste0(var,"_c")] <- taurspibda[,var] - mean(taurspibda[,var], na.rm=T)
}

# all scans for individuals in baseline tau sample
colnames(taurspibda)[colnames(taurspibda)=="rs_blsavi"] <- "rs_blsavi_base"
taurspibda_long <- merge(taurspibda, rsda, by=c("blsaid"))
taurspibda_long$interval <- difftime(taurspibda_long$tau_date, taurspibda_long$rs_date, units="days") # interval in days between tau and rs scans
taurspibda_long <- taurspibda_long[order(taurspibda_long$blsaid,taurspibda_long$rs_blsavi),]

# save spreadsheet with all participants in sample
write.xlsx(taurspibda, file.path(datapath,'tau_rs_sample.xlsx'), row.names=F)
write.xlsx(taurspibda_long, file.path(datapath,'taurs_long_sample.xlsx'), row.names=F)