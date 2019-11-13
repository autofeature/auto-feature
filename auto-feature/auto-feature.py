#!/usr/bin/env python
# coding: utf-8

# In[1]:


if __name__ == "__main__":
    
 class autofeature:

        import pandas
        import numpy as np
        import math
        import sys
        import time


        # In[2]:


        def TnI_binary(bin_test,missing_threshold=0.01):
            '''
            Checks for columns with binary values, exlcuding nan and converts the column into dummy variables; output returned includes 
            bin_test: dataframe whose binary variables are to be converted
            missing_threshold: Conversion happens only if the number of missing values are greater than threshold percentage of total number of rows    
            Why...above a certain threshold?
            '''
            
            bin_cols=bin_test.columns
            for cols in bin_cols:
                if len(bin_test[cols].dropna().unique())==2:
                    if sum(bin_test[cols].isna())>len(bin_test[cols])*missing_threshold:
                        bin_test[cols].fillna('missing',inplace=True)
                        dummy_temp=pandas.get_dummies(bin_test[cols],prefix=cols,dummy_na=False)
                        bin_test=pandas.concat([bin_test,dummy_temp],axis=1)
            return bin_test         


        # In[3]:


        def TnI_cat(testdata, x, y, missing_flag = True, missing_thres = 0.5, min_bin_size = 0.1,             binary_var = True,bv_min_size = 0.1, bv_min_incindex = 0.1, bar = False):
            
            
            #Check if the dataset contains the column name
            if x not in testdata.columns: sys.exit('ERROR!!! x variable not found in dataset, please check variable name')
            if y not in testdata.columns: sys.exit('ERROR!!! y variable not found in dataset, please check variable name')
            

            #Create duplicate columns 
            testdata['x_royalias']=testdata[x]
            testdata['y_royalias']=testdata[y]
            
            #Exit with error for drop if missing values are more than a threshold percentage
            if sum(testdata['x_royalias'].isna())/len(testdata)>missing_thres: sys.exit('ERROR!!! x variable has more than '+str(round(missing_thres*100,1))+'% missing, please consider drop')
            
            #Exit with error if the column data type is not string
            if pandas.api.types.is_numeric_dtype(testdata['x_royalias']): sys.exit('ERROR!!! '+x+' variable is not string type. TnI_categorical requires x variable to be string, please check variable type')
            
            #Exit with error for drop if there is any missing value in y
            if sum(testdata['y_royalias'].isna())>0: sys.exit('ERROR!!! y (' +y+') variable has missing value. TnI_categorical needs y variable to be fully populated. Please consider eliminating missing y values')
            
            #Exit with error if the datatype for y is a string
            if pandas.api.types.is_string_dtype(testdata['y_royalias']): sys.exit('ERROR!!! y ('+y+') variable is not numeric or integer. TnI_categorical requires y variable to be numeric or integer, please check variable type')
            
            #Check if the variable has potential to be ids column
            if testdata['x_royalias'].nunique()>1000: sys.exit('ERROR!!! x variable is a character variable with more than 1000 levels, highly likely to be ID vars')

            #build lookup table with only one row and build a column indicating missing entries in x variable and also rename that column to x_missing_flag 
            lookup_table=pandas.DataFrame({'var_name':x,'original_name':x,'type':'original'},index=[0])

            if missing_flag==True:
                if testdata['x_royalias'].isna().mean()>0: # do we need mean here or count would do?
                    testdata['x_missing_flag']=[1 if ct==True else 0 for ct in list(pandas.isnull(testdata['x_royalias']))]
                    lookup_table=pandas.concat([lookup_table,pandas.DataFrame({'var_name':x+'_missing_flag','original_name':x,'type':'missing_flag'},index=[1])])
                    testdata=testdata.rename(columns={'x_missing_flag':x+'_missing_flag'})
                else:
                    if bar in [False]:
                        print('there is no missing in independent variable, no inputing was done, no missing flag was created')


            #based on y_index to re-assign value
            testdata['x_royalias'].fillna('TnImissing',inplace=True) # to avoid issue when joining by this var
            testlookup=testdata[['y_royalias','x_royalias']].groupby(['x_royalias']).mean().reset_index()
            testlookup=testlookup.rename(columns={'y_royalias':'y_royindex1'})
            testlookup=testlookup.sort_values(by='y_royindex1',ascending=True)
            testlookup['x_royassign']=list(range(1,(len(testlookup)+1)))
            testlookup.drop(['y_royindex1'],axis=1,inplace=True)
            testlookup2=testlookup.copy()
            testlookup2['x_royalias'].replace('TnImissing',numpy.nan,inplace=True)
            lookup_table_rep=pandas.DataFrame({'original_name':numpy.repeat(x,len(testlookup2)),'old_val':testlookup2['x_royalias'],'new_val':testlookup2['x_royassign']})


            #append back to testdata
            testdata=testdata.merge(testlookup,how='left',on='x_royalias')

            ## rename alias for lookup
            testdata['x_royalias'].replace('TnImissing',numpy.nan,inplace=True)
            testlookup=testlookup.rename(columns={'x_royalias':x}) ######Two columns with same name created...should I delete one?
            new_column_temp_name=x+'_TnI_assign'
            testlookup=testlookup.rename(columns={'x_royassign':new_column_temp_name})
            del new_column_temp_name
            
            
            ## bin the re-assigned value
            #Ignore divide by 0 warning in NumPy - globally
            numpy.seterr(divide='ignore')

            if bar == False:
                print('min_bin_size = '+str(min_bin_size))

            #Define later after building tab below
            if len(testlookup)==1:
                if bar ==False:
                    print('With bounds, only one unique value')

            else:
                counter1=0
                counter2=0
                tab=testdata['x_royassign'].value_counts().sort_index()
                cut_point=numpy.full((len(tab)-1), numpy.nan)

                for ii in range(0,(len(tab)-1)):
                    counter1=tab.iloc[ii]
                    counter2=counter1+counter2
                    if counter2/sum(testdata['x_royassign'].isna())>=min_bin_size:
                        cut_point[ii]=int(tab.index[ii])
                        counter2=0
                del counter1,counter2,ii
                cut_point=numpy.array([x for x in list(cut_point) if x is not numpy.nan])
                if sum(testdata['x_royassign']>cut_point[-1])/sum(testdata['x_royassign'].notnull())<min_bin_size:
                    cut_point=cut_point[:-1]
            

            cut_point=[-numpy.inf]+list(cut_point)+[numpy.inf]

            testdata['x_roybins_bi']=pandas.cut(testdata['x_royassign'],cut_point, right=True)
            testdata['x_roybins_bi']=testdata['x_roybins_bi'].astype(str)

            ## recreate y index for bins
            temp_df=testdata.groupby(['x_roybins_bi'])[['y_royalias']].mean().reset_index()
            temp_df=temp_df.rename(columns={'x_roybins_bi':'x_roybins_bi','y_royalias':'y_royindex'})
            testdata=testdata.merge(temp_df,how='left',left_on='x_roybins_bi',right_on='x_roybins_bi')


            ## take binary vars
            if binary_var==True:
                temp_x_roybins_bi=testdata['x_roybins_bi']
                testdata=pandas.get_dummies(testdata,columns=['x_roybins_bi'])
                testdata['x_roybins_bi']=temp_x_roybins_bi
                del temp_x_roybins_bi
            
                drop2=[] 
            
                for ii in range((len(testdata.columns)-len(testdata['x_roybins_bi'].unique())-1),(len(testdata.columns)-1)):
                    if testdata.iloc[:,ii].mean()<bv_min_size:
                        drop2.append(ii)
                    else:
                        if abs(testdata['y_royindex'][testdata.iloc[:,ii]==1].mean()/testdata['y_royindex'].mean()-1)<bv_min_incindex:
                            drop2.append(ii)
            
                if len(drop2)==0:
                    if bar == False:
                        print('all binary variable taken are significant')
                else:
                    if bar == False:
                        print(str(len(testdata['x_roybins_bi'].unique())-len(drop2))+ ' significant binary variable was taken')
                        testdata=testdata.drop(list(testdata.columns[drop2]),axis='columns')
                del drop2, ii
            
            ## rename alias
            lookup_table=lookup_table.append({'var_name':(x+'_TnI_assign'), 'original_name':x,'type':'assign'}, ignore_index=True)
            testdata=testdata.rename(columns={'x_royassign':(x+'_TnI_assign')})
            lookup_table=lookup_table.append({'var_name':(x+'_assign_bins'), 'original_name':x,'type':'assign_bins'}, ignore_index=True)
            testdata=testdata.rename(columns={'x_roybins_bi':(x+'_assign_bins')})
            idx=[testdata.columns.get_loc(cols) for cols in testdata.columns if cols.startswith('x_roybins')]
            lookup_table_bin=pandas.DataFrame()

            temp_upper_bound=[]
            temp_lower_bound=[]
            temp_var_name=[]

            for ii in idx:
                temp_lower_bound.append(float(testdata.columns[ii].split(',')[0].split('(')[1]))
                temp_upper_bound.append(float(testdata.columns[ii].split(',')[1].replace(']','').replace(' ','')))
                temp_var_name.append(testdata.columns[ii].replace('x_roybins',x))
                lookup_table=lookup_table.append({'var_name':testdata.columns[ii].replace('x_roybins_bi_',(x+'_bi_')), 'original_name':x,'type':'categorical binary'}, ignore_index=True)
            
            lookup_table_bin['var_name']=temp_var_name
            lookup_table_bin['lower_bound']=temp_lower_bound
            lookup_table_bin['upper_bound']=temp_upper_bound
            testdata.columns=[cols.replace('x_roybins_bi_',(x+'_bi_')) for cols in testdata.columns]
            lookup_table=lookup_table.append({'var_name':(x+'_dependent_var_index'), 'original_name':x,'type':'dependent_var_index'}, ignore_index=True)
            testdata=testdata.rename(columns={'y_royindex':(x+'_dependent_var_index')})
            testdata=testdata.drop(['x_royalias','y_royalias'],axis='columns')

            testlist={'data':testdata,'lookup':{'lookup_table':lookup_table, 'lookup_table_bin':lookup_table_bin, 'lookup_table_rep':lookup_table_rep}}
            
            return testlist


        # In[4]:


        def quantile_function():
            print('need to build this function to find quantile based on Inverse of empirical distribution function')


        # In[5]:


        def TnI_cont(testdata, x, y, missing_flag = True, missing_thres = 0.5, min_bin_size = 0.1, method = 'index',
                     binary_var = True, bv_min_size = 0.1, bv_min_incindex = 0.1, other_transform = True, bounds = .98,
                     transformations = ['Inv', 'Sqrt', 'Exp', 'Pw2', 'Log'], bar = False):

            ## check typo - #Check if the dataset contains the column name
            if x not in testdata.columns: sys.exit('ERROR!!! x variable not found in dataset, please check variable name')
            if y not in testdata.columns: sys.exit('ERROR!!! y variable not found in dataset, please check variable name')
            
            #Create duplicate columns 
            testdata['x_royalias']=testdata[x]
            testdata['y_royalias']=testdata[y]

            ## Error message
            #Exit with error for drop if missing values are more than a threshold percentage
            if sum(testdata['x_royalias'].isna())/len(testdata)>missing_thres: sys.exit('ERROR!!! x variable has more than '+str(round(missing_thres*100,1))+'% missing, please consider drop')

            #Exit with error if the column data type is not string
            if pandas.api.types.is_string_dtype(testdata['x_royalias']): sys.exit('ERROR!!! '+x+' variable is not numeric. TnI_categorical requires x variable to be numeric, please check variable type')
            
            #Exit with error for drop if there is any missing value in y
            if sum(testdata['y_royalias'].isna())>0: sys.exit('ERROR!!! y (' +y+') variable has missing value. TnI_categorical needs y variable to be fully populated. Please consider eliminating missing y values')
            
            #Exit with error if the datatype for y is a string
            if pandas.api.types.is_string_dtype(testdata['y_royalias']): sys.exit('ERROR!!! y ('+y+') variable is not numeric or integer. TnI_categorical requires y variable to be numeric or integer, please check variable type')
            
            if bar==False:
                print('method=index') if not method else print('method: '+method)## Why does bar = False; seem to be confusing    

            if method not in ['index','index_med','mean', 'median']:
                sys.exit('ERROR!!! method variable needs to be index, mean or median')


            ## missing flag
            if (missing_flag == True):
                testdata['x_missing_flag']=[1 if ct==True else 0 for ct in list(pandas.isnull(testdata['x_royalias']))]

            ## core binning part
            if bar==False:
                print('min_bin_size = ',min_bin_size)

            #THIS NEEDS TO BE FIXED
            #bnds <- quantile_function(testdata$x_royalias, c(.5 - bounds/2, .5 + bounds/2), type = 1, na.rm = T)
            bnds=[1,7]

            lookup_table_bin=pandas.DataFrame({'var_name':x,'lower_bound':bnds[0],'upper_bound':bnds[1]},index=[0])
            testdata['x_royalias']=pandas.Series(numpy.minimum(numpy.maximum(testdata['x_royalias'].to_numpy(),bnds[0]),bnds[1])) ## THIS NEEDS TO BE VERIFIED

            if(bnds[0] == bnds[1]):
                if bar ==False:
                    print('With bounds, only one unique value')

            else:
                tab=testdata['x_royalias'].value_counts(sort=False)
                n=len(testdata['x_royalias'].isna())
                i=0
                j=0
                cut_point=[]
            
                b=tab[0:(len(tab)-1)].cumsum()/n
            
                while i<1:
                    a=list(b[b>min_bin_size+i].index)
                    if len(a)>=1:
                        cut_point.insert(j,a[0])  ##  THIS COULD CAUSE ISSUES ON DIFFERENT DATASET
                        i=b[a[0]]
                        j=j+1
                    else:
                        i=1
                        j=j-1
                    
                cut_point=cut_point[0:j]

            if bar==False:
                print('bin boundaries listed below')

            cut_point=[-numpy.inf]+cut_point+[bnds[1]]
            if bar==False:print(cut_point)
            
            
            testdata['x_roybins_bi']=pandas.cut(testdata['x_royalias'],cut_point, right=True)
            testdata['x_roybins_bi_factor']=testdata['x_roybins_bi']
            testdata['x_roybins_bi']=testdata['x_roybins_bi'].astype(str)


            #Merge mean with original dataset
            testdata=testdata.merge(testdata.groupby(['x_roybins_bi'])[['y_royalias']].mean().reset_index().rename(columns={'y_royalias':'y_royindex'}),how='left',on='x_roybins_bi')

            ## missing input and other transform part
            testdata['y_roydistance']=abs(testdata['y_royindex']-testdata['y_royindex'][testdata['x_royalias'].isna()].mean())
            if testdata['x_royalias'].isna().mean()>0:
                if method=='index':
                    lookup_table_rep=pandas.DataFrame({'original_name':x,'old_val':numpy.NaN,'new_val':testdata['x_royalias'][testdata['y_roydistance']==testdata['y_roydistance'][testdata['y_roydistance']>0].min()].mean()},index=[0])
                    testdata['x_royalias'][testdata['x_royalias'].isna()]=testdata['x_royalias'][testdata['y_roydistance']==testdata['y_roydistance'][testdata['y_roydistance']>0].min()].mean()
                    if bar==False:
                        print('missing imputed by dependent var index based on bins. Imput value: ',testdata['x_royalias'][testdata['y_roydistance']==testdata['y_roydistance'][testdata['y_roydistance']>0].min()].mean())
            
                if method=='index_med':
                    lookup_table_rep=pandas.DataFrame({'original_name':x,'old_val':numpy.NaN,'new_val':quantile_function(testdata['x_royalias'][testdata['y_roydistance']==testdata['y_roydistance'][testdata['y_roydistance']>0].min()],.5)},index=[0])
                    testdata['x_royalias'][testdata['x_royalias'].isna()]=quantile_function(testdata['x_royalias'][testdata['y_roydistance']==testdata['y_roydistance'][testdata['y_roydistance']>0].min()],.5)
                    if bar==False:
                        print('missing imputed by dependent var index based on bins. Imput value: ',quantile_function(testdata['x_royalias'][testdata['y_roydistance']==testdata['y_roydistance'][testdata['y_roydistance']>0].min()],.5))
            
                if method=='mean':
                    lookup_table_rep=pandas.DataFrame({'original_name':x,'old_val':numpy.NaN,'new_val':testdata['x_royalias'].mean()},index=[0])
                    testdata['x_royalias'][testdata['x_royalias'].isna()]=testdata['x_royalias'].mean()
                    if bar==False:
                        print('missing imputed by dependent var index based on bins. Imput value: ',testdata['x_royalias'].mean())
            
                if method=='median':
                    lookup_table_rep=pandas.DataFrame({'original_name':x,'old_val':numpy.NaN,'new_val':testdata['x_royalias'].median()},index=[0])
                    testdata['x_royalias'][testdata['x_royalias'].isna()]=testdata['x_royalias'].median()
                    if bar==False:
                        print('missing imputed by dependent var index based on bins. Imput value: ',testdata['x_royalias'].median())
            
                testdata['x_roybins_bi'][testdata['x_roybins_bi'].isna()]='missing'
                drop1=['y_roydistance','y_royalias']
            
            else:
                if bar == False:
                    print('there is no missing in independent variable, no inputing was done, no missing flag was created')
                if 'med' in method:
                    lookup_table_rep=pandas.DataFrame({'original_name':x,'old_val':numpy.NaN,'new_val':testdata['x_royalias'].median()},index=[0])
                else:
                    lookup_table_rep=pandas.DataFrame({'original_name':x,'old_val':numpy.NaN,'new_val':testdata['x_royalias'].mean()},index=[0])
                drop1=['y_roydistance','y_royalias','x_royalias','x_missing_flag']

            ## take binary var part
            if binary_var==True:
                temp_x_roybins_bi=testdata['x_roybins_bi']
                testdata=pandas.get_dummies(testdata,columns=['x_roybins_bi'])
                testdata['x_roybins_bi']=temp_x_roybins_bi
                del temp_x_roybins_bi
            
                drop2=[] 
            
                for ii in range((len(testdata.columns)-len(testdata['x_roybins_bi'].unique())-1),(len(testdata.columns)-1)):
                    if testdata.iloc[:,ii].mean()<bv_min_size:
                        drop2.append(ii)
                    else:
                        if abs(testdata['y_royindex'][testdata.iloc[:,ii]==1].mean()/testdata['y_royindex'].mean()-1)<bv_min_incindex:
                            drop2.append(ii)
            
                if len(drop2)==0:
                    if bar == False:
                        print('all binary variable taken are significant')
                else:
                    if bar == False:
                        print(str(len(testdata['x_roybins_bi'].unique())-len(drop2))+ ' significant binary variable was taken')
                        testdata=testdata.drop(list(testdata.columns[drop2]),axis='columns')
                del drop2, ii
            
            testdata=testdata[[col for col in testdata.columns if col not in ['x_roybins_bi']]]

            lookup_table=pandas.DataFrame({'var_name':x, 'original_name':x, 'type':'original'},index=[0])

            ## other transform part
            if other_transform==True:
                if bar ==False:
                    temp_string=""
                    for i in range(0,len(transformations)):
                        if i ==0:
                            temp_string=temp_string+transformations[i]
                        else:
                            temp_string=temp_string+', '+transformations[i]
                        
                    print('continuous var transformation: '+temp_string)
                    del temp_string
                
                if 'Pw2' in transformations:
                    testdata['x_roytran_TnI_pw2']=testdata['x_royalias']**2
                    lookup_table=lookup_table.append({'var_name':(x+'_TnI_pw2'),'original_name': x, 'type':'pw2'},ignore_index=True)
            
                if str(0) not in testdata['x_royalias'] and 'Inv' in transformations:
                    testdata['x_roytran_TnI_inv']=1/testdata['x_royalias']
                    lookup_table=lookup_table.append({'var_name':(x+'_TnI_inv'),'original_name': x, 'type':'inv'},ignore_index=True)
            
                if bnds[0]>0 and 'Sqrt' in transformations:
                    testdata['x_roytran_TnI_sqrt']=testdata['x_royalias']**.5
                    lookup_table=lookup_table.append({'var_name':(x+'_TnI_pw2'),'original_name': x, 'type':'sqrt'},ignore_index=True)


                if bnds[0]>0 and 'Log' in transformations:
                    testdata['x_roytran_TnI_log']=numpy.log(testdata['x_royalias'])
                    lookup_table=lookup_table.append({'var_name':(x+'_TnI_log'),'original_name': x, 'type':'log'},ignore_index=True)
                

                if 'Exp' in transformations:
                    testdata['x_roytran_TnI_exp']=numpy.exp((testdata['x_royalias']-testdata['x_royalias'].min())/(testdata['x_royalias'].max()-testdata['x_royalias'].min()))
                    lookup_table=lookup_table.append({'var_name':(x+'_TnI_exp'),'original_name': x, 'type':'exp'},ignore_index=True)
            
                testdata.columns=[c if 'x_roytran' not in c else c.replace('x_roytran',x) for c in testdata.columns]
            
            
            ## rename alias
            lookup_table=lookup_table.append({'var_name':(x+'_bins'),'original_name': x, 'type':'bins'},ignore_index=True)
            testdata=testdata.rename(columns={'x_roybins_bi_factor':(x+'_bins')})


            idx=[testdata.columns.get_loc(cols) for cols in testdata.columns if cols.startswith('x_roybins')]

            for ii in idx:
                rge = [testdata['x_royalias'][testdata.iloc[:,ii]==1].min(),testdata['x_royalias'][testdata.iloc[:,ii]==1].max()]
                if rge[0]==bnds[0]:
                    up=-numpy.Inf
                else:
                    up=testdata['x_royalias'][(testdata.iloc[:,14]==0) & (testdata['x_royalias']<rge[0])].max()
            
                lookup_table_bin=lookup_table_bin.append({'var_name':testdata.columns[ii].replace('x_roybins',x),'lower_bound':up,'upper_bound':rge[1]},ignore_index=True)
                del rge, up

            lookup_table=lookup_table.append({'var_name':(x+'_TnI_processed'), 'original_name':x, 'type':'TnI_processed'},ignore_index=True)
            testdata=testdata.rename(columns={'x_royalias':(x+'_TnI_processed')})

            if testdata['x_missing_flag'].mean()>0:
                print('I was here fine.....')
                lookup_table=lookup_table({'var_name': (x+'_missing_flag'),'original_name':x,'type':'missing_flag'},ignore_index=True)
                testdata=testdata.rename(columns={'x_missing_flag':(x+'_missing_flag')})

            for ii in idx:
                lookup_table=lookup_table.append({'var_name':testdata.columns[ii].replace('x_roybins',x), 'original_name':x,'type':'continuous binary'}, ignore_index=True)

            testdata.columns=[c.replace('x_roybins',x) if 'x_roybins' in c else c for c in testdata.columns]
            
            lookup_table=lookup_table.append({'var_name':(x+'_dependent_var_index'),'original_name':x,'type':'dependent_var_index'},ignore_index=True)

            testdata.columns=[c.replace('y_royindex',(x+'_dependent_var_index')) if 'y_royindex' in c else c for c in testdata.columns]

            testdata=testdata[[c for c in testdata.columns if c not in drop1]]
            del drop1

            testlist={'data':testdata,'lookup':{'lookup_table':lookup_table,'lookup_table_bin':lookup_table_bin,'lookup_table_rep':lookup_table_rep}}
            
            return testlist


        # In[6]:


        def progressbar(it, prefix="", size=60, file=sys.stdout):

            count = len(it)
            def show(j):
                x = int(size*j/count)
                file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
                file.flush()        
            show(0)
            for i, item in enumerate(it):
                yield item
                show(i+1)
            file.write("\n")
            file.flush()


        # In[7]:


        def TnI_smart(test_df, var_list, y, binary_min_threshold = 0.05, missing_flag = True, min_bin_size = 0.1,
                      method = 'index', binary_var = True, bv_min_size = 0.1, bv_min_incindex = 0.1, missing_thres = 0.5,
                      other_transform = True, bounds = .98, transformations = ['Inv', 'Sqrt', 'Exp', 'Pw2', 'Log'], bar = True):
            '''this function assumes that the y variable is numeric'''

            #Check if dataframe is provided
            if not isinstance(test_df,pandas.DataFrame): sys.exit('ERROR!!! df is not a pandas dataframe, please please convert data into a dataframe')
            dv=test_df[y]

            #pull out var lists by datatype
            int_var=var_list[var_list['data_type'] == "numeric"]
            binary_var2=var_list[var_list['data_type'] == "binary"]
            category_var=var_list[var_list['data_type'] == "categorical"] 

            #create var sets by datatype
            int_var_names=list(pandas.unique(int_var['var_name']))
            binary_var_names=list(pandas.unique(binary_var2['var_name']))
            category_var_names=list(pandas.unique(category_var['var_name']))
            other_names=[c for c in list(var_list['var_name']) if c not in (int_var_names+binary_var_names+category_var_names)]

            #for each datatype list we need to pull out there corresponding variables
            binary_df=test_df[binary_var_names]
            int_df=test_df[int_var_names]
            #int_df=int_df[,c(which(colnames(int_df)==y), which(colnames(int_df)!=y))] --- this line seems to be unnecessary
            cat_df=test_df[category_var_names]
            cat_df[y]=dv
            other_df=test_df[other_names]

            a=int_df.apply(lambda x: x.isna()).apply(lambda x: x.mean())
            b=cat_df.apply(lambda x: x.isna()).apply(lambda x: x.mean())
            many_missing=list(a[a>missing_thres].index)+list(b[b>missing_thres].index)
            del a,b

            if len(many_missing) > 0:
                sys.exit('ERROR!!! Variable(s): '+many_missing+' has more than'+missing_thres+' missing, please consider drop')

            a_temp=cat_df.apply(lambda x: len(x.value_counts())>1000)
            many_cat=list(a_temp[a_temp==True].index)
            del a_temp

            if len(many_cat)>0:
                sys.exit('ERROR!!! '+', '.join(many_cat)+' variable is a character variable with more than 1000 levels, highly likely to be ID vars')

            if len(binary_var_names)>0:
                binary_tni=TnI_binary(binary_df,binary_min_threshold)

            else:
                binary_tni=binary_df
            
            if len(int_var_names)>0:
                lookup_table_cont={}
                lookup_table_cont_bin={}
                lookup_table_cont_rep={}
                if bar == False:
                    for cols in [cols for cols in int_var_names if cols not in y]:
                        print(cols+' variable transformation starting \n')
                        tni_cont_test=TnI_cont(int_df[[y,cols]],x=cols,y=y,missing_flag=missing_flag,missing_thres=missing_thres,
                                               min_bin_size=min_bin_size,method=method,binary_var=binary_var,bv_min_size=bv_min_size,
                                               bv_min_incindex=bv_min_incindex,other_transform=other_transform,bounds=bounds,
                                               transformations=transformations,bar=bar)
                        int_df=int_df.join(tni_cont_test['data'].drop([y,cols],axis=1))
                        lookup_table_cont[cols]=tni_cont_test['lookup']['lookup_table']
                        lookup_table_cont_bin[cols]=tni_cont_test['lookup']['lookup_table_bin']
                        lookup_table_cont_rep[cols]=tni_cont_test['lookup']['lookup_table_rep']
                else:
                    for cols in progressbar([cols for cols in int_var_names if cols not in y], prefix="Computing: ", size=50):
                        time.sleep(0.1)
                        tni_cont_test=TnI_cont(int_df[[y,cols]],x=cols,y=y,missing_flag=missing_flag,missing_thres=missing_thres,
                                              min_bin_size=min_bin_size,method=method,binary_var=binary_var,bv_min_size=bv_min_size,
                                              bv_min_incindex=bv_min_incindex,other_transform=other_transform,bounds=bounds,
                                              transformations=transformations,bar=bar)
                        int_df=int_df.join(tni_cont_test['data'].drop([y,cols],axis=1))
                        lookup_table_cont[cols]=tni_cont_test['lookup']['lookup_table']
                        lookup_table_cont_bin[cols]=tni_cont_test['lookup']['lookup_table_bin']
                        lookup_table_cont_rep[cols]=tni_cont_test['lookup']['lookup_table_rep']

                    lookup_table_cont=pandas.concat(lookup_table_cont.values(), ignore_index=True)
                    lookup_table_cont_bin=pandas.concat(lookup_table_cont_bin.values(), ignore_index=True)
                    lookup_table_cont_rep=pandas.concat(lookup_table_cont_rep.values(), ignore_index=True)

            else:
                lookup_table_cont=pandas.DataFrame()
                lookup_table_cont_bin=pandas.DataFrame()
                lookup_table_cont_rep=pandas.DataFrame()
            
            if len(category_var_names)>0:
                lookup_table_cat={}
                lookup_table_cat_bin={}
                lookup_table_cat_rep={}
                if bar == False:
                    for cols in [cols for cols in category_var_names if cols not in y]:
                        print(cols+' variable transformation starting \n')
                        tni_cat_test=TnI_cat(cat_df[[y,cols]],x=cols,y=y,missing_flag=missing_flag,missing_thres=missing_thres,
                                             min_bin_size=min_bin_size,binary_var=binary_var,bv_min_size=bv_min_size,
                                             bv_min_incindex=bv_min_incindex,bar=bar)            
                    
                        cat_df=cat_df.join(tni_cat_test['data'].drop([y,cols],axis=1))
                        lookup_table_cat[cols]=tni_cat_test['lookup']['lookup_table']
                        lookup_table_cat_bin[cols]=tni_cat_test['lookup']['lookup_table_bin']
                        lookup_table_cat_rep[cols]=tni_cat_test['lookup']['lookup_table_rep']
                
                else:
                    for cols in progressbar([cols for cols in category_var_names if cols not in y], prefix="Computing: ", size=50):
                        time.sleep(0.1)
                        tni_cat_test=TnI_cat(cat_df[[y,cols]],x=cols,y=y,missing_flag=missing_flag,missing_thres=missing_thres,
                                             min_bin_size=min_bin_size,binary_var=binary_var,bv_min_size=bv_min_size,
                                             bv_min_incindex=bv_min_incindex,bar=bar)
                        cat_df=cat_df.join(tni_cat_test['data'].drop([y,cols],axis=1))
                        lookup_table_cat[cols]=tni_cat_test['lookup']['lookup_table']
                        lookup_table_cat_bin[cols]=tni_cat_test['lookup']['lookup_table_bin']
                        lookup_table_cat_rep[cols]=tni_cat_test['lookup']['lookup_table_rep']

                    lookup_table_cat=pandas.concat(lookup_table_cat.values(), ignore_index=True)
                    lookup_table_cat_bin=pandas.concat(lookup_table_cat_bin.values(), ignore_index=True)
                    lookup_table_cat_rep=pandas.concat(lookup_table_cat_rep.values(), ignore_index=True)

            else:
                lookup_table_cat=pandas.DataFrame()
                lookup_table_cat_bin=pandas.DataFrame()
                lookup_table_cat_rep=pandas.DataFrame()
            
            
            cat_df=cat_df.drop([y],axis=1)
            smart_df=pandas.concat([other_df, int_df, cat_df, binary_tni],ignore_index=True,axis=1)
            smart_df.columns=list(other_df.columns)+list(int_df.columns)+list(cat_df.columns)+list(binary_tni.columns)
            lookup_table=pandas.concat([lookup_table_cont,lookup_table_cat], ignore_index=True)
            lookup_table_rep=pandas.concat([lookup_table_cont_rep,lookup_table_cat_rep], ignore_index=True)
            lookup_table_bin=pandas.concat([lookup_table_cont_bin,lookup_table_cat_bin], ignore_index=True)
            smart_misingflags=smart_df.filter(regex='_missing_flag')
            smart_dependent=smart_df.filter(regex='_dependent_var_index')
            smart_bins=smart_df.filter(regex='_bins')
            smart_assign=smart_df.filter(regex='_TnI_assign')

            temp_columns=smart_df.filter(regex='_TnI_assign|_missing_flag|_dependent_var_index|_bins')
            smart_predictors=smart_df[[cols for cols in smart_df.columns if cols not in temp_columns]]
            del temp_columns

            drop_original_columns=[cols for cols in category_var_names if cols not in y]+[cols for cols in int_var_names if cols not in y]+[cols for cols in other_names if cols not in y]
            smart_predictors=smart_predictors[[cols for cols in smart_predictors if cols not in drop_original_columns]]

            testlist={'smart_predictors':smart_predictors,'smart_all':smart_df,'smart_assign':smart_assign,'smart_bins': smart_bins,
                      'smart_dependent':smart_dependent,'smart_misingflags':smart_misingflags,
                      'lookup':{'lookup_table':lookup_table,'lookup_table_bin':lookup_table_bin,'lookup_table_rep':lookup_table_rep}}

            return testlist


        # In[ ]:




