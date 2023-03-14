
# loading libraries
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.conf import SparkConf
from awsglue.context import GlueContext
from pyspark.context import SparkContext
import re
import boto3
import pandas as pd
import os
from botocore.exceptions import ClientError
import numpy as np
from rapidfuzz.fuzz import QRatio
import unicodedata
from RecordLinkageModel import *
import math
# passing environment variables arguments
args = getResolvedOptions(sys.argv, ['bucket', 'file', 'path','matches','cencia_clean_file_path','entrypoint','fuzzy_match','model_matches','matching_output'])


class S3transformations:
    # init variables and arguments
    def __init__(self):
        self.model_mathes_path = args['model_matches']
        self.entrypoint = args['entrypoint']
        self.file = args['file']
        self.bucket = args['bucket']
        self.fuzzy_match = args['fuzzy_match']
        self.company_matches_path = args['matches']
        self.cencia_clean_file_path = args['cencia_clean_file_path']
        self.path = args['path']
        self.matching_output = args['matching_output']
        self.new_datapoint = []
        self.session = boto3.Session()
        self.s3 = self.session.resource('s3')
        self.client = self.session.client('s3')
        self.dataframes = []
        self.df = pd.DataFrame()
        self.df2 = pd.DataFrame()
        self.null_data = pd.DataFrame()
        self.match_data = pd.DataFrame()
        self.df1 = pd.DataFrame()
        self.conf = SparkConf()
        self.aux_df = pd.DataFrame()
        self.msj = self.conf.set("spark.rpc.message.maxSize", "2000").set('spark.executor.memory','12g').set("spark.driver.memory",'12g').\
        set("spark.executor.memoryOverhead","15300").set("spark.defaul.parallelism","10")
        self.context = SparkContext(conf=self.msj)
        self.glueContext = GlueContext(self.context)
        self.spark_session = self.glueContext.spark_session
        

    def load_file(self):
        try:
            # loading files  from the path argument
            file = self.file.split(",")
            # verifying inf bucket exists
            for bucket_ in self.s3.buckets.all():
                if self.bucket in bucket_.name:
                    # loading bucket and files
                    bucket = self.s3.Bucket(self.bucket)
                    for obj in bucket.objects.filter(Prefix="Company_Match/"):
                        for item in file:
                            if item in obj.key:
                                # reading the data from the files
                                obj = self.s3.Object(bucket_name=self.bucket, key=self.path + item)
                                response = obj.get()['Body'].read().decode('ISO-8859-1').splitlines(True)
                                # inserting the data into one dataframe
                                self.df = pd.DataFrame(response)
            #updating col name
            self.df = self.df.rename(columns={0: "company_name"})
            self.aux_df["original_cencia_name"] = self.df["company_name"] # added
            #removing noise from df
            self.df = self.cleaning_data(self.df,"company_name",1)
            self.df = self.wrangling_df(self.df) 
            self.df = self.df.rename(columns={"index":"id"})
            loc_id = [x for x in self.df.id]
            self.aux_df = self.aux_df.loc[loc_id,["original_cencia_name"]].reset_index(drop=True)
            self.df = self.df.drop(columns=["id"])
            self.aux_df = pd.concat([self.df,self.aux_df],axis=1)
            self.df['source'] ='HCDL - CENSIA' 
            self.save_bucket_file(self.df,self.cencia_clean_file_path, "clustered_file.csv")
            return  self.df
        except ClientError as e:
            print("the dataframe could not be created", e)
            
            
    @classmethod 
    # regex rules to clean de censia file dataset
    def cleaning_data(cls,*args,**kwargs):
        try:
            if args[0] is None or  args[1] is None or args[2] is None :
                print("arguments not passed")
            else:
                if args[2] == 1:
                    df = args[0]
                    for col in df.columns:
                        if args[1] in col:
                            df[col] =  df[col].apply(lambda x :  unicodedata.normalize('NFD', x).encode('ascii', 'ignore').decode()) #removing accents
                            df[col] = df[col].apply(lambda x : re.sub(r'([^\s\w,.])|(^[\s\b\s]+)|([\s\b\s]+$)|([^A-Z0-9,.\s])|(^[,]+)|([,]+$)','',x))#remove non alphanumeric data
                            df[col] = df[col].apply(lambda x : re.sub(r'(^[\d\b\d]+$)|(^[\d]*[^\w][\d]*)','',x)) #remove noisy digits repetitions
                            df[col] = df[col].apply(lambda x : re.sub(r'(^[,.]+)|([,.]+$)','',x)) #remove dots and commas repetitions and the end of beginning of the word 
                            df[col] = df[col].apply(lambda x : re.sub(r'(^[\s\b.]+)','',x))  #replace several noisy spaces at the beginning fo the word with dots
                            df[col] =  df[col].apply(lambda x: re.sub(r'(^[\s\b\s]+)|([\s\b\s]+$)', '', x)) #replace several noisy spaces at the end or beginning fo the word
                            df[col] =  df[col].apply(lambda x: re.sub(r'([0-9]*[\W][0-9]*[\w][\W][0-9]*)|(^[0-9]*$)|(^[\s]+[0-9]*$)', '', x)) #Replace more than digits repetition with ''
                            df[col] =  df[col].apply(lambda x: re.sub(r'[+\-]?[^A-Za-z]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', '', x))  #Replace scientific format (2.00e2|-2.00E34) with ''
                            df[col] =  df[col].apply(lambda x: re.sub(r'([^\w][,]+$)', '', x))  #Replace commas repetitions between words with ''
                            df[col] =  df[col].apply(lambda x : re.sub(r'[\s](.)\1*$','' ,x))  #Replace letter repetition with space at the start (\saaaa) with ''
                            df[col] =  df[col].apply(lambda x : re.sub(r'^(.)\1+[\s]','' ,x))  #Replace letter repetition with space at the end (aaaa\s) with ''
                            df[col] =  df[col].apply(lambda x : re.sub(r'([^A-Z0-9,.\s])','',x)) #Replace not A\N with ''
                            df[col] =  df[col].apply(lambda x : re.sub(r'(^[\s])','',x)) #Replace a string that start with space [\s] with ''
                            df[col] =  df[col].apply(lambda x : re.sub(r'(^CORPORATION)|(^[\s][CORPORATION]+$)','',x)) #remove only the word corporation
                            df[col] =  df[col].apply(lambda x : re.sub(r'(?:[.,]+[\s])',' ',x)) #replace commas and dots between spaces with space
                            df[col] =  df[col].apply(lambda x : re.sub(r'\b[(,)]\b|\b[(.)]\b',' ',x)) #replace commas or dots between spaces with ''
                            df[col] =  df[col].apply(lambda x : re.sub(r'(?:[.]+$)',' ',x)) #replace dot at the end of the words with ''
                            df[col] =  df[col].apply(lambda x : re.sub(r'((?<=[\s])[LTD]+$)|((?<=[\s])\bLTD\b)','LIMITED',x)) #replace LTD with LIMITED
                            df[col] =  df[col].replace("", np.nan) #Replace  '' with np.nan
                            return df
                    
            #option 2 applying this rules for the fuzzy matching string logic for better results
            if not kwargs['dataframes']:
                print("arguments not passed")
            else:
                payload = kwargs['dataframes']
                data = []
                shape = len(payload["datasets"])
                for i in range(0,shape):
                    for col in payload["datasets"][i]:
                        if payload["col"] in col and  payload["val"]==2:
                            payload["datasets"][i][col]=  payload["datasets"][i][col].apply(lambda x : str(x) if isinstance(x,float)  else x) #reseting columns with str values
                            payload["datasets"][i]["original_companyName"] =   payload["datasets"][i][col]
                            payload["datasets"][i][col] = payload["datasets"][i][col].apply(lambda x : unicodedata.normalize('NFD', x).encode('ascii', 'ignore').decode()) #removing accents
                            payload["datasets"][i][col] = payload["datasets"][i][col].apply(lambda x : re.sub(r'([^A-Za-z0-9\s])',' ',x)) #removing no alphanumeric data
                            payload["datasets"][i][col]=  payload["datasets"][i][col].apply(lambda x : x.upper() if x.lower()  else x) #converting words to uppercase
                    data.append(payload["datasets"][i])
                return data     
        except Exception as e:
            print("some error ocurred transforming the data, please review", e)
    
    def wrangling_df(self, aux_df):
        try:
            df = aux_df.copy() # added
            # exploring columns with noise and null values
            for col in df.columns:
                df[col] = df.loc[:, col].apply(lambda x: np.nan if x == np.inf or x == -np.inf else x)
                self.null_data = df.loc[:, df.columns].apply(lambda x: x.isna().sum()).sum()
            # extracting the columns with noise and null values
            null_values = {'null_values': self.null_data, 'columns': [col for col in df.columns]}
            # looping through the columns with null values
            for key in null_values.keys():
                # if the key columns with null values exist then drop the null values
                if 'columns' in key:
                    # dropping null values
                    df = df.dropna(subset=null_values[key]).reset_index(drop=False) # Modified, Original Value = True
                    # deleting duplicated data 
                    if df[df.duplicated(keep=False)].shape[0] > 0:
                        cols = [col for col in df.columns]
                        df = df.loc[:, cols].drop_duplicates().reset_index(drop=False) # Modified, Original Value = True
                        return df
                    # if duplicated data does not exist return dataframe with no changes
                    if df[df.duplicated(keep=False)].shape[0] == 0:
                        return df
        except Exception as e:
            print("the dataframe wrangling operation could not be created", e)
            
  
    def save_bucket_file(self,df,path,file):
        try:
            # capturing the s3 bucket
            response = self.s3.meta.client.head_bucket(Bucket=self.bucket)
            # uploading the data into the bucket if the bucket exist 
            if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                df.to_csv(path + file, index=False, encoding='utf-8')
            # printing error if the bucket does not exist
            if response['ResponseMetadata']['HTTPStatusCode'] == 404:
                print("Bucket Does Not Exist!", self.bucket)
        except ClientError as e:
            print("the file could not be loaded, please review bucket permissions o create the bucket", e)
            
            
    @classmethod    
    def matching_process(cls,threshold,df1,df2,col1,col2,col3):
        try:
            # temp variales
            matches = []
            names =[]
            source = []
            df1_temp = df1.copy()
            df2_temp = df2.copy()
            df2_temp[col2]= df2_temp[col2].astype(str)
            df1_temp[col1]= df1_temp[col1].astype(str)
            list_values = list(df2_temp[col2].unique())
            cleaned_data = list(df1_temp[col1].unique())
            source_data  = list(df1_temp[col3].unique())[0]
            # validating QRatio using threshold=85
            for item in cleaned_data:
                for value in list_values:
                    match = QRatio(item,value)
                    if  match >= threshold:
                        matches.append(item)
                        names.append(value)
                        source.append(source_data)
            # appeding all the matches in two diferentes dataframes
            companies_matches = pd.Series(matches).rename('company_matches')
            names_matches = pd.Series(names).rename('names_matches')
            source_matches  = pd.Series(source).rename('source_matches')
            df = pd.concat([companies_matches,names_matches,source_matches], axis=1)
            return df
        except Exception as e:
            print("the data can not be processed, some error ocurred",e)
            
    @classmethod
    def df_chunks(cls,df,df1):
        try:
            # datasets balancing
            # creating same shape for unbalanced dataset for do the partitioning process
            def chunking_rows_transformations(temp,temp_1):
                
                if temp.shape[0] > temp_1.shape[0]:
                    constant = math.ceil(temp.shape[0] / temp_1.shape[0])
                    temp_1 = pd.concat([temp_1]*constant)
                    temp_1 = temp_1.reset_index(drop=True)
                    temp_1 = temp_1.loc[0:temp.shape[0]-1]
                    return temp,temp_1
                if temp_1.shape[0] > temp.shape[0]: 
                    constant = math.ceil(temp_1.shape[0] / temp.shape[0])
                    temp = pd.concat([temp]*constant)
                    temp = temp.reset_index(drop=True)
                    temp = temp.loc[0:temp_1.shape[0]-1]
                    return temp,temp_1
                if temp_1.shape[0] == temp.shape[0]:
                    return temp,temp_1
                    
            #creating the chunks for the fuzzy logic.    
            df_temp, df_temp_0 = chunking_rows_transformations(df,df1)
            matches = []
            start = 0
            counter =0
            chunksize = math.ceil(df_temp.shape[0] / 10) # 20
            step = math.ceil(df_temp.shape[0]/chunksize)
            for i in range(0,step):
                start = counter
                finish = start + chunksize
                counter = counter + chunksize 
                df2 = df_temp.loc[start:finish]
                df1 = df_temp_0.loc[start:finish]
                df_matches = S3transformations.matching_process(85,df2,df1,'company_name','company_name','source')
                matches.append(df_matches)
            return matches
        except Exception as e:
            print("the data can't be loaded please review the process",e)
   

    # function to apply the fuzzy string logic
    def fuzzy_string_comparision(self):
        try:
            # generating dynamic value for cpaiq dataset
            def dynamic_data(df):
                try:
                    var_df = ["df"+str(i+1) for i in range(len(df))]
                    for var in var_df:
                        for dataset in df:
                            exec(var + '=dataset')
                            datapoint = vars()[var]
                        return datapoint
                except Exception as e:
                    print('can not retrieve dynamic data, please review the process',e)
                    
            # renaming and assigning new columns to the  df
            def renaming_cols(data,aux,company_match,names_match,source_match,company_name,columns_list):
                try:
                    df = pd.concat(data).reset_index(drop=True)
                    df = df.rename(columns=company_match)
                    df = df.rename(columns=names_match)
                    df = df.rename(columns=source_match)
                    aux = aux.rename(columns=company_name)
                    aux=  aux[columns_list]
                    return df, aux
                except Exception as e:
                    print('can not create dataframe output,please review the process',e )
            
            
            # cleaning dataset columns    
            def clean_columns(df):
                try:
                    pattern = r'[\s]+|[^A-Za-z0-9]'
                    match = re.compile(pattern)
                    result_match = df.columns.str.match(match)
                    if True in result_match:
                        df.columns = df.columns.str.replace('[\s]','',regex=True)
                        df.columns = df.columns.str.replace('[^A-Za-z0-9]','', regex=True)
                        return df
                    else:
                        return df
                except Exception as e:
                    print('can not retrieve data',e)
                    
            # reindex datasets list for censia df after cleaning     
            def extract_source(df_list,source):
                try:
                    aux_list = df_list.copy()
                    index_df = []
                    for index, df in enumerate(aux_list):
                        entry_df = df.loc[(df.source == source),df.columns]
                        if entry_df.equals(df) is True:
                            index_df.append(aux_list[index])
                            del aux_list[index]
                    return aux_list,index_df
                except Exception as e:
                    print('can not retrieve data',e)

            
            companies_files = []
            bucket = self.s3.Bucket(self.bucket)
            # bucket path for reading files
            for obj in bucket.objects.filter(Prefix=f"Company_Match/datasets/"):
                if obj.key.endswith(".csv"):
                    path = "s3://"+self.bucket+"/"+ obj.key
                    # reading all the files in folder
                    df = pd.read_csv(path, sep=',', encoding='latin-1',low_memory=False)#low_memory=False
                    df = clean_columns(df)
                    companies_files.append(df)
                    #companies_files.append(pd.read_csv(path, sep=',', encoding='latin-1'))
            # creating a dict with all datasets values from the files
            data = {"datasets":companies_files,"col":"company_name","val":2}
            # applying some regex rules to the datasets
            datapoints = self.cleaning_data(None,None,None, dataframes=data)
            
            # reading cpaiq data from folder and applying some regex rules
            entrypoint = pd.read_csv(self.entrypoint ,sep=',')
            #entrypoint = entrypoint.loc[0:2000]
            new_entrypoint = {"datasets":[entrypoint],"col":"company_name","val":2}
            cleaned_entrypoint = self.cleaning_data(None,None,None, dataframes=new_entrypoint)
            # calling for dynamic assignament
            df1 = dynamic_data(cleaned_entrypoint)
            
            # starting  the chunking process for applying fuzzy matching string logic for company name comparision
            chunks_parts = []
            for dataset in datapoints:
                chunks = self.df_chunks(dataset,df1)
                chunks_parts.append(chunks)

            # adding the clean cencia dataset block    
            aux_datapoints, index_cencia_data = extract_source(datapoints,'HCDL - CENSIA')    
            index_cencia_data[0] = pd.merge(index_cencia_data[0],self.aux_df, on=['company_name'], how='inner')
            index_cencia_data[0] = index_cencia_data[0].drop(columns=['original_companyName']).rename(columns={'original_cencia_name':'original_companyName'})
            new_datapoint = [index_cencia_data[0]]
            
            # process to merge   with the original cencia company name column with the others datasets
            self.new_datapoint = new_datapoint + aux_datapoints
            
            #merge operation resulting of the fuzzy logic with the originals datasets
            aux_data = []
            data = []
            for datapoint,temp_data in zip(self.new_datapoint,chunks_parts):
                df,aux_df = renaming_cols(temp_data,datapoint,
                                            {'company_matches':'COMPANY_MATCH'},
                                            {'names_matches':'CPAIQ_MATCH'},
                                            {'source_matches':'source'},
                                            {"company_name":"CPAIQ_MATCH"},
                                            ["CPAIQ_MATCH","original_companyName"])
                aux_data.append(aux_df)
                data.append(df)
            
            #saving the datasets into the bucket after the merging operation
            counter = 0    
            for temp_1, temp_2 in zip(aux_data,data):
                df_match=pd.merge(temp_2,temp_1, on=["CPAIQ_MATCH"], how="inner").drop_duplicates().reset_index(drop=True)
                df_match = df_match.rename(columns={'source':'SOURCE'})
                df_match.to_csv(self.fuzzy_match + 'company_matches_'+str(counter)+'.csv',index=False)
                counter = counter + 1
            
            #datasets info
            return self.new_datapoint
        except Exception  as e:
            print('can not create the datasets comparision results, review the process',e)
       
    
    @classmethod
    # applying the recordlinkage model
    def apply_model(cls,df,col_1,col_2,col_3,col_4,col_5,col_6,col_7):
        try:
            df1_a = cleaning_datasets(df,col_1)
            df1_a1, df1_b1 = temp_index(df1_a,col_2,col_1,col_3)
            df1_c1 = df_comparision(df1_a1, df1_b1,col_1,col_1,col_4,col_5)
            true_links_1 = true_links(df1_c1,col_5)
            result_svm, golden_matches_test_index = svm_classifier(df1_c1,true_links_1)
            df_f1 = matching_index_results(df1_c1,result_svm)
            df_f1 = matching_data(df_f1,col_6, df1_a1,col_1)
            df_f1 = matching_data(df_f1,col_7, df1_b1,col_4)
            eval_fscore(true_links_1,result_svm)
            matrix_1 = confusion_matrix(df1_c1,golden_matches_test_index,result_svm)
            df_f1 = matching_original_values(df_f1,df1_a1)
            return df_f1
        except Exception as e:
            print("the results of the model can not be apply, please review the process", e)# merging datasets for final output (model output)
    
    
    @classmethod
    # merge operations to  extract addional data such as : original company name and source data
    def df_output(cls,df,aux_df,label1,label2,label3,label4,label5,label6,label7,label8,label9,label10,label11):
        try:
            df = df.rename(columns={label1:label2}).drop(columns=[label3])
            df = df.rename(columns={label2:label4})
            aux_df =aux_df.rename(columns={label3:label2}) 
            aux_df = aux_df[[label2,label5,label11]] 
            df3 = aux_df.merge(df, left_on=label2, right_on=label4 , how="inner").drop_duplicates().reset_index(drop=True)
            df3 =df3.drop(columns=[label2])
            df3 = df3.rename(columns={label6:label7,label5:label8})
            df3 = df3.rename(columns={label8:label9})
            df3 = df3.drop(columns=[label10])
            return df3
        except Exception as e:
            print("can not merge the datasets, please review the process", e)

    # datasets cast operations
    # cast all the columns with in32 and float32 values
    @classmethod
    def columns_cast(cls,df):
        try:
            float_col = [col for col in df.columns if df[col].dtypes == "float64"]
            int_col = [col for col in df.columns if df[col].dtypes == "int64"]
            for col in float_col:
                df[col] = df[col].astype("float32")
            for col in int_col:
                df[col] = df[col].astype("int32")
            return df
        except Exception as e:
            print("can not cast the dataset columns, please review the process", e)

    def generate_model_output(self):
        try:
            chunks = []
            bucket = self.s3.Bucket(self.bucket)
            # bucket path for reading files
            for obj in bucket.objects.filter(Prefix=f"Company_Match/matches/"):
                if obj.key.endswith('.csv'):
                    path = "s3://"+self.bucket+"/"+ obj.key
                    chunksize = 100000
                    for df in pd.read_csv(path,chunksize=chunksize,encoding='latin-1'):
                        aux_df1 = df.copy()
                        df = df[['CPAIQ_MATCH','COMPANY_MATCH','SOURCE']]
                        # apply the model
                        df_1 = self.apply_model(df,"CPAIQ_MATCH","id","ORIGINAL_NAMES_COMPANIES","COMPANY_MATCH","CPAIQ_MATCH_SCORE","id_1","id_2")
                        # merging the final output in both dataframes
                        df_2 = self.df_output(df_1,aux_df1,"ORIGINAL_NAMES_COMPANIES","CPAIQ_DATA","CPAIQ_MATCH","CPAIQ_DATA_MATCH","original_companyName",
                        "id","ID","ORIGINAL_COMPANY_NAME","CPAIQ_ORIGINAL_NAME","id_2","SOURCE")
                        df_2 = self.columns_cast(df_2)
                        chunks.append(df_2)
                        self.df2 = pd.concat(chunks).reset_index(drop=True)
            source_key = [s_key for s_key in self.df2.SOURCE.unique()]
            for i, s_key in enumerate(source_key):
                output = self.df2.loc[(self.df2.SOURCE == s_key),self.df2.columns].reset_index(drop=True)
                output.to_csv(self.model_mathes_path+"model_matches"+"_"+str(i)+".csv",index=False)
        except Exception as e:
            print('could not create the final output, please review the process',e)
            
            
    def source_matches(self):
        try:
            #applying regex rules for all the rows in datasets
            # if we need to add more rules just we need to add more keys to the dictionary
            def apply_regex_rules(df,col,source_col):
                try:
                    # dict with the regex rules 4 and 5
                    regex_dict = {
                    4:{lambda x: x.upper()}, # to upper values
                    5:{lambda x: x.replace('(^\s+|\s+$)|([^0-9a-zA-Z\s]+)','',regex=True)} # remove non ap data and rtrim and ltrim spaces
                    }
                    lb_functions = []
                    for val in regex_dict.values():
                        lb_functions.append(val)
                    datasets = []
                    df[col] = df[source_col]
                    for cols in df.columns:
                        if 'VALUES' in cols:
                            index_col = 'VALUES'
                            for index,item in enumerate(lb_functions):
                                df[index_col] = df[index_col].apply(lb_functions[index])
                                data = pd.DataFrame(df[index_col])
                                datasets.append(data)
                    regex_data = pd.concat(datasets).reset_index(drop=True)
                    start = 0
                    keys = [key for key in regex_dict.keys()]
                    loc_indexer = regex_data.shape[0]/len(datasets)
                    for i in range(len(keys)):
                        regex_data.loc[start:loc_indexer,['REGEX_RULE']] = keys[i]
                        start = start + loc_indexer
                        loc_indexer = loc_indexer * 2
                        
                    df = df.drop(columns=[col])
                    df = pd.concat([df]*len(datasets)).reset_index(drop=True)
                    final_output  = pd.concat([df,regex_data],axis=1)
                    final_output['REGEX_RULE'] = final_output['REGEX_RULE'].apply(lambda x: int(x))
                    final_output['CLUSTER_ID'] = final_output.apply(lambda row:  str(row['REGEX_RULE'])+str(row['CLUSTER_ID']), axis=1)
                    final_output = final_output.drop(columns=['REGEX_RULE'])
                    return final_output
                except Exception as e:
                    print('can not retrieve dataset, please review the process',e)
            
            #function that applies additional cleaning to the dataset and drop unnecesary columns and
            def index_columns(df):
                try:
                    df_columns = [x for x in df.columns]
                    if 'original_companyName' in df_columns:
                        for i, item in enumerate(df_columns):
                            if 'original_companyName' in df_columns[i] or 'companyId' in  df_columns[i]:
                                pos = i
                                df_columns.pop(pos)
                        return df_columns
                except Exception as e:
                    print('can not retrieve index columns, please review the process',e)
            
            # applying regex rules for the temp column of CENSIA
            def apply_regex(df,col,regex_col):
                try:
                    bool_val =  (df.loc[:, col] == 'HCDL - CENSIA').any()
                    if bool_val == True:
                        df[regex_col] = df[regex_col].apply(lambda x : re.sub(r'[\n]','',x))
                        df[regex_col] = df[regex_col].apply(lambda x : re.sub(r'[^A-Z0-9\s,.]','',x))
                        return df
                    else:
                        return df
                except Exception as e:
                    print('can not retrieve transformed dataset, please review the process',e)
                
            # function that generates the cluster_id value and filters 100% matches
            def generate_cluster_id(df):
                try:
                    df = self.columns_cast(df)
                    df['CLUSTER_ID'] = df['CPAIQ_ORIGINAL_NAME'].rank(method='dense', ascending=False)
                    df = df.sort_values(by=['CLUSTER_ID']).reset_index(drop=True)
                    df['CPAIQ_MATCH_SCORE']= df['CPAIQ_MATCH_SCORE'].apply(lambda x : x *100)
                    target = [x for x in df.columns if df[x].dtypes=='int64' or df[x].dtypes=='float64']
                    for col in target:
                        df[col] = df[col].apply(lambda x : int(x))
                    df = df.loc[(df['CPAIQ_MATCH_SCORE']== 100), df.columns].drop(columns=['ID']).rename(columns={'CPAIQ_MATCH_SCORE':'PERCENTAGE_MATCH_CPAIQ'}).\
                    reset_index(drop=True)
                    return df
                except Exception as e:
                    print('can not save the dataframe into the bucket, please review the process',e)
                    
            def df_drop(df1,df2):
                try:
                    df = pd.merge(df1,df2, left_on=['company_name'],right_on=['COMPANY_MATCH'],how='inner').drop_duplicates()
                    dfcolumns = index_columns(df1)
                    df = df.drop(columns=dfcolumns).reset_index(drop=True)
                    df = apply_regex(df,'SOURCE','original_companyName')
                    return df
                except Exception as e:
                    print('the output can not be generated, please review the process',e)
                    
            # reindex original company values after merge the final output.        
            def evaluate_rows(df):
                try:
                    if (df.loc[:,"ORIGINAL_COMPANYNAME"].isna()).any() == True:
                        df1 = pd.DataFrame(df.loc[:,"ORIGINAL_COMPANYNAME"].isna())
                        df1 = df1[df1["ORIGINAL_COMPANYNAME"]] == True
                        df1 = df1.reset_index(drop=False).rename(columns={"index":"LOC_INDEXER"})
                        loc_indexer = [x for x in df1.LOC_INDEXER]
                        for loc_index in loc_indexer:
                            df.loc[loc_index:loc_index,["ORIGINAL_COMPANYNAME"]] = df.loc[loc_index:loc_index,"ORIGINAL_COMPANY_NAME"]
                        return df
                except Exception as e:
                    print('can not reindex values, please verify the process', e)

            data = []
            input_data = []
            bucket = self.s3.Bucket(self.bucket)
            # bucket path for reading files
            for obj in bucket.objects.filter(Prefix=f"Company_Match/model_matches/"):
                if obj.key.endswith('.csv'):
                    path = "s3://"+self.bucket+"/"+ obj.key
                    df =pd.read_csv(path,encoding='latin-1')
                    data.append(df)
           
            for temp_data, df_data in zip(self.new_datapoint, data):
                df = df_drop(temp_data, df_data)
                input_data.append(df)
                
            # joins operations to create the final output
            datasets = pd.concat(input_data).reset_index(drop=True)
            output = generate_cluster_id(datasets)
            output.columns = output.columns.str.upper()
            output = output.drop(columns=['CPAIQ_DATA_MATCH','COMPANY_MATCH'])
            cpaiq_data = pd.read_csv(self.entrypoint ,sep=',')
            cpaiq_data = cpaiq_data.drop_duplicates()
            cpaiq_data = cpaiq_data.rename(columns={'companyId':'COMPANYID','company_name':'CPAIQ_ORIGINAL_NAME','source':'SOURCE'})
            final_df = pd.merge(cpaiq_data,output, on=['CPAIQ_ORIGINAL_NAME'],how='inner')
            final_df = final_df.drop(columns=['ORIGINAL_COMPANYNAME','SOURCE_y','COMPANYID_y'])
            final_df = final_df.rename(columns={'COMPANYID_x':'COMPANYID','SOURCE_x':'SOURCE','CPAIQ_ORIGINAL_NAME':'ORIGINAL_COMPANY_NAME'})
            final_df = pd.concat([final_df,output]).drop_duplicates()
            final_df = final_df.sort_values(by=['CLUSTER_ID']).drop(columns=['CPAIQ_ORIGINAL_NAME']).reset_index(drop=True)
            final_df['COMPANYID'] = final_df['COMPANYID'].apply(lambda x: str(x) if isinstance(x,float) or isinstance(x,int) and not np.isnan(x) else x)
            final_df = final_df.rename(columns={'COMPANYID':'COMPANY_ID'})
            final_df = self.columns_cast(final_df)
            final_df =evaluate_rows(final_df)
            final_df = final_df.drop(columns=['ORIGINAL_COMPANY_NAME']).rename(columns={'ORIGINAL_COMPANYNAME':'ORIGINAL_COMPANY_NAME'})
            final_df = final_df.loc[:,['CLUSTER_ID','SOURCE','COMPANY_ID','ORIGINAL_COMPANY_NAME','PERCENTAGE_MATCH_CPAIQ']].drop_duplicates().reset_index(drop=True)
            final_df = apply_regex_rules(final_df,'VALUES','ORIGINAL_COMPANY_NAME')
            final_df.to_csv(self.matching_output + 'model_output.csv',index=False)
        except Exception as e:
            print('the final output file can not be generated, please review the process',e)
        
      
if __name__ == "__main__":
    s3 = S3transformations()
    s3.load_file()
    s3.fuzzy_string_comparision()
    s3.generate_model_output()
    s3.source_matches()
    
    
   
    
    
    