
.libPaths(c("/proj/cae_muc/q521100/83_Miniconda/r4.0.5/library/", .libPaths()))

env_windows <- FALSE

path_split <- "/"

##########################
# START
##########################
print(paste("[CEOELA] ELA computation starts.", sep=""))

# deactivate warning
options(warn=-1)

# import all library
library("readxl")
library("flacco")
library("openxlsx")
library("rjson")
library("rstudioapi")
library("parallel")
library("mda")
library("class")
library("mlr")
library("ParamHelpers")
library("tibble")

########################## 
# get working directory for R session
# set to TRUE, if running in windows
# set to FALSE, if running in unix
# by default TRUE
if (env_windows) {
  # set to TRUE, if running script directly in Rstudio
  # set to FALSE, if calling script via rpy2 in Python
  # by default FALSE
  if (FALSE) {
    # Rstudio 
    path_script <- getSourceEditorContext()$path
    current_wd <- dirname(path_script)
  } else {
    # rpy2
    current_wd <- dirname(sys.frame(1)$ofile)
  }
} else {
  current_wd <- getwd()
}
setwd(current_wd)  


##########################
# META-DATA
##########################
# read meta-data
json_file <- paste(current_wd, '/CEOELA_metadata.json', sep="")
json_data <- fromJSON(file=json_file)

# all available features in flacco
#list_feature <- list("cm_angle", "cm_conv", "cm_grad", "ela_conv", "ela_curv", "ela_distr", "ela_level", 
#                     "ela_local", "ela_meta", "basic", "disp", "limo", "nbc", "pca", "bt", "gcm", "ic")

# all features considered
list_feature_base <- list("ela_distr", "ela_level", "ela_meta", "basic", "disp", "limo", "nbc", "pca", "ic")
list_feature_crash <- list_feature_base
list_feature_BBOB <- list_feature_base
list_feature_AF <- list_feature_base

problem_label <- json_data$problem_label
filepath_save <- json_data$filepath_save
list_input <- json_data$list_input
list_output <- json_data$list_output
bootstrap <- json_data$bootstrap
bootstrap_size <- json_data$bootstrap_size
bootstrap_repeat <- json_data$bootstrap_repeat
BBOB_func <- json_data$BBOB_func
BBOB_instance <- json_data$BBOB_instance
AF_number <- json_data$AF_number
ELA_problem <- json_data$ELA_problem
ELA_BBOB <- json_data$ELA_BBOB
ELA_AF <- json_data$ELA_AF
np_ela <- json_data$np_ela

numCores <- detectCores()
ncore <- as.integer(min(c(np_ela, numCores)))


##########################
# compute_ELA
##########################
# compute ELA features with flacco
compute_ELA <- function(data, x_data, list_ELA, obj){
  
  y_data <- unlist(data[obj])
  y_data <- unname(y_data)
  feat.object = createFeatureObject(X=x_data, y=y_data)
  
  # calculate all features
  list_ELA_value = list()
  for (feature_i in 1:length(list_ELA)){
    feat_error <- TRUE
    tryCatch({
      featureSet_temp <- calculateFeatureSet(feat.object, set=unlist(list_ELA[feature_i]))
      feat_error <- FALSE
    }, error=function(e){cat("ERROR :", conditionMessage(e), "\n")})
    
    # skip if error in feature computation
    if (feat_error){
      list_ELA_value[[feature_i]] <- NULL
      next
    }
    
    df_featvalue_temp <- data.frame(matrix(unlist(featureSet_temp), nrow=length(featureSet_temp), byrow=TRUE))
    colnames(df_featvalue_temp) <- obj
    df_featname_temp <- data.frame(matrix(unlist(names(featureSet_temp)), nrow=length(featureSet_temp), byrow=TRUE))
    colnames(df_featname_temp) <- c("ELA_feat")
    df_feature_temp <- dplyr::bind_cols(df_featvalue_temp, df_featname_temp)
    list_ELA_value[[feature_i]] <- df_feature_temp
  } # END FOR
  
  # combine feature result
  df_feature_main <- dplyr::bind_rows(list_ELA_value)
  print(paste("Objective ", obj, " done!", sep=""))
  return(df_feature_main)
}


##########################
# compute_ELA_mp
##########################
# multiprocessing for ELA features computation
compute_ELA_mp <- function(data, x_data, list_ELA, list_obj){
  result_mp <- mclapply(list_obj, function(obj_name) {
    compute_ELA(data, x_data, list_ELA, obj=obj_name)
  }, mc.cores=ncore)
  return(result_mp)
}


##########################
# func_extract_LA_feat
##########################
# combine ELA features over all objectives
func_extract_LA_feat <- function(data, list_dv, list_ELA, list_obj){
  x_data <- data.matrix(data[list_dv])
  result_ELA <- compute_ELA_mp(data, x_data, list_ELA, list_obj)
  
  first_run <- TRUE
  for (obj_i in 1:length(list_obj)){
    df_feature_temp <- result_ELA[[obj_i]]
    df_feature_temp <- data.frame(df_feature_temp, row.names=2)
    if (first_run){
      df_feature <- df_feature_temp
      first_run <- FALSE
    } else {
      df_feature <- merge(df_feature, df_feature_temp, by="row.names", all=TRUE)
      df_feature <- data.frame(df_feature, row.names=1)
    }
  }
  df_feature <- tibble::rownames_to_column(df_feature, "ELA_feat")
  return(df_feature)
}


##########################
# BASE FUNCTION
##########################
# Base function to call ELA function
func_base <- function(prob_type, list_sheet, list_input, list_obj, list_ELA){
  print(paste("[CEOELA] ELA ", prob_type, " is running...", sep=""))
  
  if (bootstrap){
    if (all(prob_type=="AF")){ # artificial functions
      for (sheet_i in 1:length(list_sheet)){
        sheetname_temp <- unlist(list_sheet[sheet_i])
        filepath <- paste(filepath_save, path_split, problem_label, '_', prob_type, '_', sheetname_temp, '.xlsx', sep="")
        df_data <- read_excel(filepath, sheet=sheetname_temp)
        
        # call function to extract LA features
        df_feature <- func_extract_LA_feat(df_data, list_input, list_ELA, list_obj)
        
        # save results
        wb <- createWorkbook()
        addWorksheet(wb, sheetname_temp)
        writeData(wb, sheetname_temp, df_feature, startRow=1, startCol=1)
        filename_temp <- paste(filepath_save, path_split, "featELA_", problem_label, "_", prob_type, '_', sheetname_temp, ".xlsx", sep='')
        saveWorkbook(wb, file=filename_temp, overwrite=TRUE)
        print(paste("[CEOELA] ", prob_type, " Sheet ", sheetname_temp, " done!", sep=""))
      }
    } else { # problem instance and BBOB
      filepath <- paste(filepath_save, path_split, problem_label, '_', prob_type, '.xlsx', sep="")
      wb <- createWorkbook()
      
      for (sheet_i in 1:length(list_sheet)){
        sheetname_temp <- unlist(list_sheet[sheet_i])
        df_data <- read_excel(filepath, sheet=sheetname_temp)
        
        # call function to extract LA features
        df_feature <- func_extract_LA_feat(df_data, list_input, list_ELA, list_obj)
        
        # save results
        addWorksheet(wb, sheetname_temp)
        writeData(wb, sheetname_temp, df_feature, startRow=1, startCol=1)
        print(paste("[CEOELA] ", prob_type, " Sheet ", sheetname_temp, " done!", sep=""))
      }
      filename_temp <- paste(filepath_save, path_split, "featELA_", problem_label, "_", prob_type, ".xlsx", sep='')
      saveWorkbook(wb, file=filename_temp, overwrite=TRUE)
    } # END IF
    
  } else { # no bootstrapping
    if (all(prob_type=="BBOB")){ # BBOB
      filepath <- paste(filepath_save, path_split, problem_label, '_', prob_type, '.xlsx', sep="")
      wb <- createWorkbook()
      
      for (sheet_i in 1:length(list_sheet)){
        sheetname_temp <- unlist(list_sheet[sheet_i])
        df_data <- read_excel(filepath, sheet=sheetname_temp)
        
        # call function to extract LA features
        df_feature <- func_extract_LA_feat(df_data, list_input, list_ELA, list_obj)
        
        # save results
        addWorksheet(wb, sheetname_temp)
        writeData(wb, sheetname_temp, df_feature, startRow=1, startCol=1)
        print(paste("[CEOELA] ", prob_type, " Sheet ", sheetname_temp, " done!", sep=""))
      }
      filename_temp <- paste(filepath_save, path_split, "featELA_", problem_label, "_", prob_type, ".xlsx", sep='')
      saveWorkbook(wb, file=filename_temp, overwrite=TRUE)
    } else { # problem instance and AF
      filepath <- paste(filepath_save, path_split, problem_label, '_', prob_type, '.xlsx', sep="")
      wb <- createWorkbook()
      df_data <- read_excel(filepath, sheet="full")
      
      # call function to extract LA features
      df_feature <- func_extract_LA_feat(df_data, list_input, list_ELA, list_obj)
      
      # save results
      sheetname_temp <- "full"
      addWorksheet(wb, sheetname_temp)
      writeData(wb, sheetname_temp, df_feature, startRow=1, startCol=1)
      print(paste("[CEOELA] ", prob_type, " Sheet ", sheetname_temp, " done!", sep=""))
      filename_temp <- paste(filepath_save, path_split, "featELA_", problem_label, "_", prob_type, ".xlsx", sep='')
      saveWorkbook(wb, file=filename_temp, overwrite=TRUE)
    }
  }
  print(paste("[CEOELA] ", prob_type, " ELA ", problem_label, " done.", sep=""))
} # END DEF





##########################
# Instance (original)
########################## 
if (ELA_problem) {
  func_base('original', bootstrap_repeat, list_input, list_output, list_feature_crash)
} # END IF



##########################
# Instance (re-scale)
########################## 
if (ELA_problem & ELA_BBOB) {
  func_base('rescale', bootstrap_repeat, list_input, list_output, list_feature_crash)
} # END IF



##########################
# BBOB Functions
########################## 
if (ELA_BBOB) {
  func_base('BBOB', BBOB_instance, list_input, BBOB_func, list_feature_BBOB)
} # END IF




##########################
# Artificial Functions
########################## 
if (ELA_AF) {
  func_base('AF', bootstrap_repeat, list_input, AF_number, list_feature_AF)
} # END IF





##########################
# END
##########################
print(paste("[CEOELA] ELA computation done!", sep=""))




