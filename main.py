import boto3
import json
import numpy as np
import multiprocessing
import time
import pandas as pd
from botocore.client import Config

config = Config(connect_timeout=1000, read_timeout=1000)
#####update the below
client = boto3.client('lambda','us-east-1',aws_access_key_id="temp",aws_secret_access_key="temp+YG7Ux",config=config)

#####update the below if lambda name is different
def lambdaCall(rate,start,end,weights):
    response = client.invoke(
    FunctionName='project',
    InvocationType='RequestResponse',
    LogType='None',
    Payload=json.dumps({"start":start,"end":end,"weights":json.dumps(weights)}))

    return json.loads(response['Payload'].read().decode())






def driver_func(lambdas,limit,weights,rate):
    PROCESSES = lambdas
    params=[]
    start=0
    for i in range(1,lambdas+1):
        temp=[]
        temp.append(rate)
        temp.append(start)
        temp.append((i*limit)//lambdas)
        start=(i*limit)//lambdas
        temp.append(weights)
        params.append(temp)
        
    with multiprocessing.Pool(PROCESSES) as pool:
#         params = [(1, ), (2, ), (3, ), (4, )]
#         results = [pool.apply_async(temp, p) for p in params]
        result = pool.starmap(lambdaCall,params)
    return result


def add2(a,b):
    if(type(a) == list):
        if(type(a[0])==list):
            for i in range(len(a)):
                add2(a[i],b[i])
        else:
            for i in range(len(a)):
                a[i]=a[i]+b[i]

def divide(a,b):
    if(type(a) == list):
        if(type(a[0])==list):
            for i in range(len(a)):
                divide(a[i],b)
        else:
            for i in range(len(a)):
                a[i]=a[i]//b
    



def combine(a,b):
    a['time']=np.add(a['time'],b['time'])
    a['summary']['loss']=np.add(a['summary']['loss'],b['summary']['loss'])
    a['summary']['acc']=np.add(a['summary']['acc'],b['summary']['acc'])
    a['summary']['sparse_categorical_accuracy']=np.add(a['summary']['sparse_categorical_accuracy'],b['summary']['sparse_categorical_accuracy'])
    add2(a['weights'],b['weights'])



if __name__ == "__main__":
    #####update the below for different configuration
    lambdas=1
    size=50000
    epochs=5
    weights=0
    run=1

    
    for epoch in range(1,epochs+1):

        
        if(epoch<50):
            rate=0.1
        else:
            rate=0.01
        start=time.time()
        res=driver_func(lambdas,size,weights,rate)
        end=time.time()
        for i in range(lambdas):
            print(res[i]['body']['summary'])
            res[i]['body']['weights']=json.loads(res[i]['body']['weights'])

        for i in range(1,lambdas):
            combine(res[0]['body'],res[i]['body'])


        loss=res[0]['body']['summary']['loss'][0]/lambdas
        acc=res[0]['body']['summary']['acc'][0]/lambdas
        sparse_categorical_accuracy=res[0]['body']['summary']['sparse_categorical_accuracy'][0]/lambdas
        billedtime=res[0]['body']['time']



        weights=res[0]['body']['weights']
        divide(weights,lambdas)



        stats=pd.DataFrame(np.array([[loss,acc,sparse_categorical_accuracy,epoch,billedtime,end-start]]),
                   columns=[ 'loss', 'accuracy', 'sparse_categorical_accuracy','epoch','billed_time','execution_time'])
        stats.to_csv("stats/"+str(epochs)+"_"+str(lambdas)+'.csv', mode='a', header=False)
        np.save("models/"+str(epoch)+"_"+str(lambdas), weights)
        print(stats)
        print("time for  epoch",epoch," is :- ",end-start," seconds")
